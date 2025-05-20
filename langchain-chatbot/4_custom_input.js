// Adds a custom `language` input to the prompt and persists it
import { START, END, StateGraph, MemorySaver, Annotation, MessagesAnnotation } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";
import { v4 as uuidv4 } from "uuid";

const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a helpful assistant. Answer in {language}."],
  ["placeholder", "{messages}"],
]);

const Schema = Annotation.Root({
  ...MessagesAnnotation.spec,
  language: Annotation(),
});

const callModel = async (state) => {
  const promptInput = await prompt.invoke(state);
  const response = await llm.invoke(promptInput);
  return { messages: [response] };
};

const workflow = new StateGraph(Schema).addNode("model", callModel).addEdge(START, "model").addEdge("model", END);

const app = workflow.compile({ checkpointer: new MemorySaver() });

const config = { configurable: { thread_id: uuidv4() } };

const result = await app.invoke(
  {
    messages: [{ role: "user", content: "What's my name?" }],
    language: "Hindi",
  },
  config
);

console.log("Custom Input Output:", result.messages.at(-1).content);
