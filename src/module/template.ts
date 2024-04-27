import { ChatOpenAI } from '@langchain/openai';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

class Template {
  private model: ChatOpenAI;
  constructor() {
    this.model = new ChatOpenAI({
      temperature: 0.5, // control the creativity of the api. If it is 0, it will be not creative at all.
      modelName: 'gpt-3.5-turbo',
    });
  }


  async main() {
    console.log('We are ready to start......');
  }

  async sampleFunction() {
    // TDDO: Implement this function
    console.log('Sample function');
  }
}

export default Template;
