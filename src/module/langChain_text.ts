import { ChatOpenAI, OpenAI, OpenAIEmbeddings } from '@langchain/openai';
import dotenv from 'dotenv';
import { RetrievalQAChain, loadQAStuffChain } from 'langchain/chains';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { Chroma } from '@langchain/community/vectorstores/chroma';

// Load environment variables
dotenv.config();

class LangChainTxT {
  private model: OpenAI;
  constructor() {
    this.model = new OpenAI({
      temperature: 0.5,
      modelName: 'gpt-3.5-turbo',
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
   * Process a text file, split it, create vector embeddings, store it in a Faiss store.
   */
  async processTextToVectorStore() {
    // Load text file
    const textLoader = new TextLoader('source_docs/openai_faq.txt');
    const textDoc = await textLoader.load();

    // Split text into paragraphs
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 200,
      chunkOverlap: 50,
    });

    const splitTextDoc = await splitter.splitDocuments(textDoc);

    const embeddings = new OpenAIEmbeddings();

    const vectorStore = await FaissStore.fromDocuments(
      splitTextDoc,
      embeddings
    );

    await vectorStore.save('./vector_store');

    console.log('Vector store saved!!!');
  }

  /**
   * Using a Faiss vector store to chatbot.
   */
  async useFaissVectorStore(question: string) {
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await FaissStore.load('./vector_store', embeddings);
    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(this.model),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true,
      verbose: false,
    });

    const res = await chain.invoke({ query: question });
  }

  /**
   * Using a ChromaDB vector store to chatbot.
   * ChromaDB is a free and open-source vector store that can be used to store and retrieve embeddings.
   *
   * @param {string} collectionName - Collection name of ChromaDB
   */
  async useChromaDBVectorStore(collectionName: string) {
    const loader = new TextLoader('source_docs/openai_faq.txt');
    const docs = await loader.load();
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 200,
      chunkOverlap: 50,
    });
    const documents = await splitter.splitDocuments(docs);

    const vectorStore = await Chroma.fromDocuments(
      documents,
      new OpenAIEmbeddings(),
      {
        collectionName,
        url: 'http://localhost:8000', // ChromaDB URL, currently setup locally
        collectionMetadata: {
          'demo:first': 'demo collection',
        },
      }
    );

    console.log('chroma vector store created', vectorStore);
  }

  /**
   * Using a ChromaDB Vector Store in a Chatbot.
   * The ChromaDB is created by useChromaDBVectorStore method.
   *
   * @param {string} prompt - The prompt to send to the model
   */
  async useChromaVectorStore(prompt: string) {
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await Chroma.fromExistingCollection(embeddings, {
      collectionName: 'langchain_demo',
      url: 'http://localhost:8000',
    });

    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(this.model),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true,
      verbose: false,
    });

    const res = await chain.invoke({ query: prompt });
  }
}

export default LangChainTxT;
