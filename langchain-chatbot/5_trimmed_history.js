// Trims conversation history to fit within token limits
import { SystemMessage, HumanMessage, AIMessage, trimMessages } from "@langchain/core/messages";

const messages = [
  new SystemMessage("You are helpful."),
  new HumanMessage("Hi, I'm Bob."),
  new AIMessage("Hi!"),
  new HumanMessage("I like pizza."),
  new AIMessage("Nice."),
  new HumanMessage("What’s 2+2?"),
  new AIMessage("4"),
  new HumanMessage("What’s my name?"),
];

const trimmer = trimMessages({
  maxTokens: 6, // using a fake token counter here for example purposes
  strategy: "last",
  tokenCounter: (msgs) => msgs.length,
  includeSystem: true,
  allowPartial: false,
});

const trimmed = await trimmer.invoke(messages);
console.log("Trimmed History:");
trimmed.forEach((msg) => console.log("-", msg.content));
