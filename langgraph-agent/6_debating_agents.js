// 6_debating_agents.js
// A team of specialized agents that debate with each other before providing a final answer

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";

// Team of agents that engage in a debate to reach the best answer
// Each agent has a different perspective and expertise

// Create LLM models for each agent
const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

// Define different agent roles for the debate
const ROLES = {
  MODERATOR: "moderator",
  CRITIC: "critic",
  CREATIVE: "creative",
  PRAGMATIC: "pragmatic",
  SUMMARIZER: "summarizer",
};

// Agent-specific system prompts to guide their behavior
const moderatorSystemPrompt = new SystemMessage(
  "You are the debate moderator. Your job is to pose the initial question clearly and guide the discussion. " +
    "Ask follow-up questions to keep the debate productive. Be neutral and focus on drawing out different perspectives. " +
    "If you feel the debate has explored the topic sufficiently, indicate that it's time to reach a conclusion by saying 'CONCLUDE_DEBATE'."
);

const criticSystemPrompt = new SystemMessage(
  "You are the critic agent. Your job is to critically analyze proposals and identify potential flaws, risks, " +
    "and unintended consequences. Be thorough in your analysis, but remain constructive. " +
    "Your goal is to strengthen ideas through thoughtful critique, not to simply reject them."
);

const creativeSystemPrompt = new SystemMessage(
  "You are the creative agent. Your job is to generate innovative ideas and approaches. " +
    "Think outside the box and propose novel solutions. Don't be constrained by conventional thinking. " +
    "Be imaginative while still addressing the core problem."
);

const pragmaticSystemPrompt = new SystemMessage(
  "You are the pragmatic agent. Your job is to focus on practical, implementable solutions. " +
    "Consider real-world constraints like resources, time, and feasibility. " +
    "Ground the discussion in reality and suggest concrete steps forward."
);

const summarizerSystemPrompt = new SystemMessage(
  "You are the summarizer agent. Your job is to synthesize the debate and form a final conclusion. " +
    "When you see 'CONCLUDE_DEBATE', carefully review all perspectives shared in the discussion. " +
    "Present a balanced final answer that incorporates the strongest points from the debate. " +
    "Begin your summary with 'FINAL_ANSWER:' followed by a comprehensive response."
);

// Function to create agent nodes
function createAgentNode(agentName) {
  let systemPrompt;

  switch (agentName) {
    case ROLES.MODERATOR:
      systemPrompt = moderatorSystemPrompt;
      break;
    case ROLES.CRITIC:
      systemPrompt = criticSystemPrompt;
      break;
    case ROLES.CREATIVE:
      systemPrompt = creativeSystemPrompt;
      break;
    case ROLES.PRAGMATIC:
      systemPrompt = pragmaticSystemPrompt;
      break;
    case ROLES.SUMMARIZER:
      systemPrompt = summarizerSystemPrompt;
      break;
    default:
      throw new Error(`Unknown agent role: ${agentName}`);
  }

  return async function ({ messages }) {
    // Add agent's name to the message for clarity in the debate
    const agentMessages = [systemPrompt, ...messages];
    const response = await llm.invoke(agentMessages);

    // Add the agent's name as a prefix to their response for clarity
    const content = `${agentName.toUpperCase()}: ${response.content}`;
    const modifiedResponse = new AIMessage({ content });

    return { messages: [modifiedResponse] };
  };
}

// Simplified routing function for the moderator
function moderatorRoute({ messages }) {
  const lastMessage = messages[messages.length - 1];
  const messageContent = lastMessage.content || "";

  // Check if the debate should conclude
  if (messageContent.includes("CONCLUDE_DEBATE")) {
    return "summarizer";
  }

  // Count how many debate turns we've had
  const agentMessages = messages.filter((m) => m instanceof AIMessage);
  const debateTurns = agentMessages.length;

  // If we've gone through multiple rounds, give the moderator a chance to conclude
  if (debateTurns >= 8) {
    return "summarizer";
  }

  return "creative";
}

// Route from creative to pragmatic
function creativeRoute() {
  return "pragmatic";
}

// Route from pragmatic to critic
function pragmaticRoute() {
  return "critic";
}

// Route from critic back to moderator
function criticRoute() {
  return "moderator";
}

// Route from summarizer to end
function summarizerRoute({ messages }) {
  const lastMessage = messages[messages.length - 1];
  const messageContent = lastMessage.content || "";

  // Check if we have a final answer
  if (messageContent.includes("FINAL_ANSWER:")) {
    return "end";
  }

  // If no FINAL_ANSWER yet, route back to moderator for another cycle
  return "moderator";
}

// Create the debate graph with simplified routing
const debateGraph = new StateGraph(MessagesAnnotation)
  .addNode(ROLES.MODERATOR, createAgentNode(ROLES.MODERATOR))
  .addNode(ROLES.CRITIC, createAgentNode(ROLES.CRITIC))
  .addNode(ROLES.CREATIVE, createAgentNode(ROLES.CREATIVE))
  .addNode(ROLES.PRAGMATIC, createAgentNode(ROLES.PRAGMATIC))
  .addNode(ROLES.SUMMARIZER, createAgentNode(ROLES.SUMMARIZER))

  // Start with the moderator
  .addEdge("__start__", ROLES.MODERATOR)

  // Define direct routing for predictable flow
  .addConditionalEdges(ROLES.MODERATOR, moderatorRoute, {
    [ROLES.CREATIVE]: ROLES.CREATIVE,
    [ROLES.SUMMARIZER]: ROLES.SUMMARIZER,
  })
  .addConditionalEdges(ROLES.CREATIVE, creativeRoute, {
    [ROLES.PRAGMATIC]: ROLES.PRAGMATIC,
  })
  .addConditionalEdges(ROLES.PRAGMATIC, pragmaticRoute, {
    [ROLES.CRITIC]: ROLES.CRITIC,
  })
  .addConditionalEdges(ROLES.CRITIC, criticRoute, {
    [ROLES.MODERATOR]: ROLES.MODERATOR,
  })
  .addConditionalEdges(ROLES.SUMMARIZER, summarizerRoute, {
    [ROLES.MODERATOR]: ROLES.MODERATOR,
    end: "__end__",
  });

// Compile the graph into a runnable application
const debateTeam = debateGraph.compile();

// Function to trace the debate as it happens
function traceDebate(messages) {
  const debateLog = [];
  for (const message of messages) {
    if (message instanceof HumanMessage) {
      debateLog.push(`HUMAN: ${message.content}`);
    } else if (message instanceof AIMessage) {
      debateLog.push(message.content);
    }
  }
  return debateLog.join("\n\n");
}

// Example usage
async function main() {
  console.log("Starting the debate...\n");

  // Debate topics to explore
  const debateTopics = [
    "What is the most effective way to address climate change?",
    "Should artificial intelligence development be regulated, and if so, how?",
    "What educational approach would best prepare students for the future?",
  ];

  for (const topic of debateTopics) {
    console.log(`\n\n=============================================`);
    console.log(`DEBATE TOPIC: ${topic}`);
    console.log(`=============================================\n`);

    const finalState = await debateTeam.invoke({
      messages: [new HumanMessage(topic)],
    });

    console.log(traceDebate(finalState.messages));

    // Extract just the final answer for a clean summary
    const lastMessage = finalState.messages[finalState.messages.length - 1];
    if (lastMessage.content.includes("FINAL_ANSWER:")) {
      const finalAnswer = lastMessage.content.split("FINAL_ANSWER:")[1].trim();
      console.log(`\n------------- SUMMARY -------------`);
      console.log(finalAnswer);
    }
  }
}

// Run the main function
main().catch(console.error);
