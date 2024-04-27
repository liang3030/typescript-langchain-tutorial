// import * as Hapi from '@hapi/hapi';
import LangchainIntro from './module/langChain_intro';
import LangChainChat from './module/langChain_chat';
import LangChainTxT from './module/langChain_text';
import ChromaDBTest from './module/ChromaDB_test';
import LangChainPDF from './module/langChain_PDF';
import LangChainURL from './module/langChain_url';

async function init() {
  const langChainUrl = new LangChainURL();
  // await chromaDBTest.processPDFToVectorStore();
  await langChainUrl.processURLToFaissVectorStore();
}

init();
