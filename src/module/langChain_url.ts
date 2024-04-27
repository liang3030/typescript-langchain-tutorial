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
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { PuppeteerWebBaseLoader } from 'langchain/document_loaders/web/puppeteer';
import * as puppeteer from 'puppeteer';
import * as cheerio from 'cheerio';
import { Document } from 'langchain/document';

// Load environment variables
dotenv.config();

class LangChainURL {
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
  async processURLToFaissVectorStore() {
    const url = 'https://simplifiedlocalgrowth.com/slg-1';
    // Load pdf file
    const loader = new PuppeteerWebBaseLoader(url, {
      launchOptions: {
        headless: true,
      },
      evaluate: async (
        page: puppeteer.Page,
        browser: puppeteer.Browser
      ): Promise<string> => {
        try {
          await page.goto(url, { waitUntil: 'networkidle0' });
          const textContent = await page.evaluate(() => {
            const bodyElement = document.querySelector('body');
            return bodyElement && bodyElement.textContent
              ? bodyElement.textContent
              : '';
          });

          await browser.close();
          return textContent ?? '';
        } catch (error) {
          console.log('Error: ', error);
          await browser.close();
          return '';
        }
      },
    });
    const urlDocs = await loader.load();

    const pageContent = urlDocs[0].pageContent;

    // load html content into cheerio
    const $ = cheerio.load(pageContent);

    $('script, style').remove(); // remove script and style tags

    const cleanedText = $('body')
      .html()
      ?.replace(/<style[^>]*>.*<\/style>/gms, '');

    // load cleaned text again to extract text
    const cleaned$ = cheerio.load(cleanedText!);
    const textContent = cleaned$('body').text();
    const docs = textContent.replace(/[^\x20-\x7E]+/g, ''); // Remove non-ASCII characters

    // Create Document instances
    const documents = [new Document({ pageContent: docs })];

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 200,
      chunkOverlap: 50,
    });

    const splitDocuments = await splitter.splitDocuments(documents);

    // Initialize OpenAI embeddings
    const embeddings = new OpenAIEmbeddings();
    // Process and store embeddings in batches
    const batchSize = 10; // Adjust based on your needs
    let allDocs = [];

    for (let i = 0; i < splitDocuments.length; i += batchSize) {
      const batch = splitDocuments.slice(i, i + batchSize);

      // Add batch documents to allDocs array
      allDocs.push(...batch);
    }

    // Load the docs into the vector store
    const vectorStore = await FaissStore.fromDocuments(allDocs, embeddings);

    await vectorStore.save('./vector-store-url');

    console.log('Faiss Vector store created successfully!');
  }

  /**
   * Using a Faiss vector store to chatbot.
   */
  async useFaissVectorStore(question: string) {
    const embeddings = new OpenAIEmbeddings();
    const vectorStore = await FaissStore.load('./vector_store_url', embeddings);

    const chain = new RetrievalQAChain({
      combineDocumentsChain: loadQAStuffChain(this.model),
      retriever: vectorStore.asRetriever(),
      returnSourceDocuments: true,
      verbose: false,
    });

    const res = await chain.invoke({
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

export default LangChainURL;
