// Using a system prompt template with placeholders
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You talk like a pirate."],
  ["placeholder", "{messages}"],
]);

const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

const formatted = await prompt.invoke({
  messages: [{ role: "user", content: "What is 2 + 2?" }],
});

const response = await llm.invoke(formatted);
console.log("Pirate Output:", response.content);
