import { OpenAI } from '@langchain/openai';
import { PromptTemplate } from '@langchain/core/prompts';
import dotenv from 'dotenv';
import { LLMChain, SimpleSequentialChain } from 'langchain/chains';
import { Calculator } from '@langchain/community/tools/calculator';
import { SearchApi } from '@langchain/community/tools/searchapi';
import { initializeAgentExecutorWithOptions } from 'langchain/agents';

// Load environment variables
dotenv.config();

class LangchainIntro {
  private model: OpenAI;
  constructor() {
    this.model = new OpenAI({
      temperature: 0.5,
      modelName: 'gpt-3.5-turbo',
    });
  }

  /**
   *
   * @param {string} prompt - The prompt to send to the model
   */
  async main(prompt: string) {
    try {
      const response = await this.model.invoke(prompt);
      console.log('Response', response);
    } catch (error) {
      console.error('Error', error);
    }
  }

  /**
   *
   * @param {string} prompt - The prompt to send to the tempate and then to the model
   */
  async promptTemplate(prompt: string) {
    const firstTemplate =
      'What could be a good project to start learning ${progammingLanguage}?';
    const promptTemplate = new PromptTemplate({
      template: firstTemplate,
      inputVariables: ['progammingLanguage'],
    });

    const formatPrompt = await promptTemplate.format({
      progammingLanguage: prompt,
    });

    console.log('Formatted prompt::', formatPrompt);

    await this.main(formatPrompt);
  }

  async promptMultipleTemplates(prompt: string) {
    // First template in the chain
    const firstTemplate =
      'What is the most popular city in ${country} for traveller. Just return the name of city?';
    const firtPromptTemplate = new PromptTemplate({
      template: firstTemplate,
      inputVariables: ['country'],
    });

    const chain1 = new LLMChain({
      llm: this.model,
      prompt: firtPromptTemplate,
    });

    // Second template in the chain
    const secondTemplate =
      'What are the top three things to do in ${city}? Just return answer as three bullet points. And display the {city} in the uppercase in above your list.';
    const secondPromptTemplate = new PromptTemplate({
      template: secondTemplate,
      inputVariables: ['city'],
    });
    const chain2 = new LLMChain({
      llm: this.model,
      prompt: secondPromptTemplate,
    });

    const overallChain = new SimpleSequentialChain({
      chains: [chain1, chain2],
      verbose: false,
    });

    try {
      const finalAnswer = await overallChain.run(prompt);
      console.log('Final answer:', finalAnswer);
    } catch (error) {
      console.error('Error:', error);
    }
  }

  async promptAgent() {
    const tools = [
      new Calculator(),
      new SearchApi(process.env.SEARCH_API, { engine: 'google' }),
    ];

    try {
      const executor = await initializeAgentExecutorWithOptions(
        tools,
        this.model,
        {
          verbose: false,
          agentType: 'structured-chat-zero-shot-react-description',
        }
      );

      console.log('agent is loaded......');

      const res = await executor.call({
        input:
          'Please tell me what is the most import event around world in 2024 until now?',
      });
      console.log(res.output);
    } catch (error) {
      console.error('Error:', error);
    }
  }

  /**
   * Demonstrates streaming responses from the model.
   * Handles real-time token-based response from the model.
   * */
  async promptStreaming() {
    const model = new OpenAI({
      temperature: 0.5,
      modelName: 'gpt-3.5-turbo',
      streaming: true,
      callbacks: [
        {
          handleLLMNewToken: (token) => {
            process.stdout.write(token);
          },
        },
      ],
    });

    await model.invoke(
      'Will Trump win the election in 2024? What is your reasons behind this answer?'
    );

    // console.log('Response:', resp);
  }
}

export default LangchainIntro;
