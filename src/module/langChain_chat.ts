import { ChatOpenAI, OpenAI } from '@langchain/openai';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from '@langchain/core/prompts';
import dotenv from 'dotenv';
import { LLMChain, SimpleSequentialChain } from 'langchain/chains';
import { Calculator } from '@langchain/community/tools/calculator';
import { SearchApi } from '@langchain/community/tools/searchapi';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

// Load environment variables
dotenv.config();

class LangChainChat {
  private model: ChatOpenAI;
  constructor() {
    this.model = new ChatOpenAI({
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
   * Makes an asynchronous call to the Open AI model with given prompt.
   * Logs response or error to the console.
   *
   * @param {string} prompt - The prompt to send to the model
   */
  async customAssistant(prompt: string) {
    try {
      await this.model.invoke([
        new SystemMessage(
          'Your name is Rico. You are a very funny guy. Answer every question with short sentences and sense of humor.'
        ),
        new HumanMessage(prompt),
      ]);
      // console.log('Response', response);
    } catch (error) {
      console.error('Error', error);
    }
  }

  async customAssistantTemplate(prompt: string, outputLanguage: string) {
    const translationPromptTemplate = ChatPromptTemplate.fromMessages([
      SystemMessagePromptTemplate.fromTemplate(
        'Your name is Juan. You are a helpful assistant to translate language from {inputLanguage} to {outputLanguage}. And answer only with {outputLanguage} prounouncation in {inputLanguage}. So that non {outputLanguage} speaker can read.'
      ),
      HumanMessagePromptTemplate.fromTemplate('{userText}'),
    ]);
    const chain = new LLMChain({
      prompt: translationPromptTemplate,
      llm: this.model,
    });

    await chain.invoke({
      inputLanguage: 'english',
      outputLanguage,
      userText: prompt,
    });
  }
}

export default LangChainChat;
