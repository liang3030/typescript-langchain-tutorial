import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from '@langchain/openai';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import dotenv from 'dotenv';
import {
  LLMChain,
  RetrievalQAChain,
  SimpleSequentialChain,
  loadQAStuffChain,
} from 'langchain/chains';
import { Calculator } from '@langchain/community/tools/calculator';
import { SearchApi } from '@langchain/community/tools/searchapi';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { Chroma } from '@langchain/community/vectorstores/chroma';

// Load environment variables
dotenv.config();

class LangChainPDF {
  private model: OpenAI;
  constructor() {
    this.model = new OpenAI({
      temperature: 0.5,
      modelName: 'gpt-4-turbo-2024-04-09',
      streaming: true,
      callbacks: [
        {
          handleLLMNewToken(token) {
            process.stdout.write(token);
          },
        },
      ],
    });
  }

  /**
   * Process a pdf file, split it, create vector embeddings, store it in a Faiss store.
   */
  async processPDFToVectorStore() {
    // Load pdf file
    const loader = new PDFLoader('source_docs/interview.pdf', {
      parsedItemSeparator: ' ',
      splitPages: false,
    });
    const textDoc = await loader.load();

    // Split text into paragraphs
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 10000,
      chunkOverlap: 50,
    });

    const splitTextDoc = await splitter.splitDocuments(textDoc);

    const embeddings = new OpenAIEmbeddings();

    const vectorStore = await FaissStore.fromDocuments(
      splitTextDoc,
      embeddings
    );

    await vectorStore.save('./vector_store_pdf');

    console.log('Vector store saved!!!');
  }

  /**
   * Using a Faiss vector store to chatbot.
   */
  async useFaissVectorStore(question: string) {
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await FaissStore.load('./vector_store_pdf', embeddings);

    // Use prompt template
    const template = `You are interviewing helper. Use the followging places of context to answer the questions. If you do not know , you can say "I do not know". 
    {context}
    Question: {question}
    `;

    const QAChainPrompt = new PromptTemplate({
      inputVariables: ['question', 'context'],
      template,
    });

    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(this.model, {
        prompt: QAChainPrompt,
      }),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true,
      verbose: false,
    });

    const res = await chain.invoke({
      systemPrompt: 'This is an interview helper. You can ask me questions.',
      query: question,
    });
  }

  /**
   * Using a ChromaDB vector store to chatbot.
   * ChromaDB is a free and open-source vector store that can be used to store and retrieve embeddings.
   *
   * @param {string} collectionName - Collection name of ChromaDB
   */
  // async useChromaDBVectorStore(collectionName: string) {
  //   const loader = new TextLoader('source_docs/openai_faq.txt');
  //   const docs = await loader.load();
  //   const splitter = new RecursiveCharacterTextSplitter({
  //     chunkSize: 200,
  //     chunkOverlap: 50,
  //   });
  //   const documents = await splitter.splitDocuments(docs);

  //   const vectorStore = await Chroma.fromDocuments(
  //     documents,
  //     new OpenAIEmbeddings(),
  //     {
  //       collectionName,
  //       url: 'http://localhost:8000', // ChromaDB URL, currently setup locally
  //       collectionMetadata: {
  //         'demo:first': 'demo collection',
  //       },
  //     }
  //   );

  //   console.log('chroma vector store created', vectorStore);
  // }

  /**
   * Using a ChromaDB Vector Store in a Chatbot.
   * The ChromaDB is created by useChromaDBVectorStore method.
   *
   * @param {string} prompt - The prompt to send to the model
   */
  // async useChromaVectorStore(prompt: string) {
  //   const embeddings = new OpenAIEmbeddings();
  //   const vectorStore = await Chroma.fromExistingCollection(embeddings, {
  //     collectionName: 'langchain_demo',
  //     url: 'http://localhost:8000',
  //   });

  //   const chain = new RetrievalQAChain({
  //     combineDocumentsChain: loadQAStuffChain(this.model),
  //     retriever: vectorStore.asRetriever(),
  //     returnSourceDocuments: true,
  //     verbose: false,
  //   });

  //   const res = await chain.invoke({ query: prompt });
  // }
}

export default LangChainPDF;
