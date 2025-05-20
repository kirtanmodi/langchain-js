// 7_rag_memory_chatbot.js
// A multi-turn chatbot with Retrieval-Augmented Generation (RAG) and conversation memory

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { formatDocumentsAsString } from "langchain/util/document";
import { Document } from "@langchain/core/documents";

// Sample knowledge base - in a real application, this would be loaded from files, databases, etc.
const sampleDocuments = [
  "LangChain is a framework for developing applications powered by language models. It enables applications that are context-aware, reason, and learn from feedback.",
  "LangGraph is an extension of LangChain that enables orchestrating agentic workflows. It's a library for building stateful, multi-actor applications with LLMs.",
  "Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by first retrieving relevant information from external sources, and then generating a response based on both the query and the retrieved information.",
  "The StateGraph class in LangGraph helps create workflows with multiple steps, where each step can be a different LLM operation, and the output of one step can influence the next steps.",
  "Vector databases store embeddings, which are numerical representations of text, images, or other data. For RAG systems, they allow efficient semantic search to find relevant information.",
  "Conversation memory in chatbots refers to the ability to maintain context across multiple interactions, allowing for more coherent and contextually appropriate responses.",
  "System prompts are instructions given to the model that guide its behavior. They are not shown to the user but help set the tone, style, and constraints for the model's responses.",
  "OpenAI's language models like GPT-4o can generate human-like text based on the input they receive. They are trained on diverse internet text but have knowledge cutoffs.",
  "Embeddings are vector representations of text that capture semantic meaning. Similar concepts have similar embeddings, which enables semantic search capabilities.",
  "Prompt engineering involves designing effective prompts to guide the behavior of language models and obtain desired outputs for specific tasks or applications.",
];

// Initialize OpenAI embeddings for vectorization
const embeddings = new OpenAIEmbeddings();

// Text splitter to break documents into chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});

// Function to initialize the vector store with sample documents
async function initializeVectorStore() {
  // Convert sample texts to Document objects
  const documents = sampleDocuments.map((text, i) => new Document({ pageContent: text, metadata: { source: `doc-${i}` } }));

  // Split documents into chunks if they're large
  const splitDocs = await textSplitter.splitDocuments(documents);

  // Create and populate the vector store
  const vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embeddings);
  return vectorStore;
}

// Main chatbot class
class RagMemoryChatbot {
  constructor() {
    this.llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });
    this.vectorStore = null;
    this.graph = null;
  }

  async initialize() {
    // Initialize the vector store with sample documents
    this.vectorStore = await initializeVectorStore();

    // Create the chatbot graph
    this.graph = this.createStateGraph();

    return this;
  }

  // Retrieval function: Get relevant documents for the current query
  async retrieveDocuments({ messages }) {
    // Extract the last user message
    const lastUserMessage = messages.filter((m) => m instanceof HumanMessage).pop();

    if (!lastUserMessage) {
      return { documents: [] };
    }

    const query = lastUserMessage.content;

    // Retrieve relevant documents based on the query
    const docs = await this.vectorStore.similaritySearch(query, 3);

    return { documents: docs };
  }

  // Document formatting function to provide context to the LLM
  formatDocuments({ messages, documents }) {
    if (!documents || documents.length === 0) {
      return { context: "No relevant information found." };
    }

    // Format the retrieved documents as a string
    const formattedDocs = formatDocumentsAsString(documents);

    return {
      context: `Relevant information from the knowledge base:\n${formattedDocs}`,
      messages,
    };
  }

  // Generate response using the LLM with retrieved context
  async generateResponse({ context, messages }) {
    // System prompt to guide the model behavior
    const systemPrompt = new SystemMessage(
      `You are a helpful AI assistant specializing in LangChain, LangGraph, and AI concepts. 
      Use the retrieved information to enhance your responses, but also use your general knowledge.
      Always be friendly, clear, and helpful. If you don't know something, say so honestly.
      
      ${context}`
    );

    // Combine system prompt with conversation history
    const chatMessages = [systemPrompt, ...messages];

    // Generate the response
    const response = await this.llm.invoke(chatMessages);

    return { messages: [response] };
  }

  // Create the StateGraph for the chatbot
  createStateGraph() {
    const workflow = new StateGraph(MessagesAnnotation)
      .addNode("retriever", this.retrieveDocuments.bind(this))
      .addNode("formatter", this.formatDocuments.bind(this))
      .addNode("generator", this.generateResponse.bind(this))

      // Define the flow: start -> retriever -> formatter -> generator -> end (for next turn)
      .addEdge("__start__", "retriever")
      .addEdge("retriever", "formatter")
      .addEdge("formatter", "generator")
      .addEdge("generator", "__end__");

    // Compile the graph
    return workflow.compile();
  }

  // Method to chat with the bot
  async chat(message, history = []) {
    // Create a new message
    const userMessage = new HumanMessage(message);

    // Add to history
    const messages = [...history, userMessage];

    // Run the graph
    const result = await this.graph.invoke({ messages });

    // Extract the AI response
    const lastMessage = result.messages[result.messages.length - 1];

    // Return the updated message history
    return [...messages, lastMessage];
  }
}

// Example usage
async function main() {
  console.log("Initializing RAG Memory Chatbot...");

  // Initialize the chatbot
  const chatbot = await new RagMemoryChatbot().initialize();

  // Sample conversation flow
  const conversation = [
    // "What is LangChain?",
    // "How does LangGraph relate to LangChain?",
    "Can you explain how RAG works?",
    // "What's the difference between embeddings and language models?",
    // "How would I implement conversation memory in a chatbot?",
  ];

  let messages = [];

  for (const userMessage of conversation) {
    console.log("\n================================================");
    console.log(`👤 User: ${userMessage}`);
    console.log("================================================\n");

    // Send message and get updated conversation
    messages = await chatbot.chat(userMessage, messages);

    // Get only the last bot response for cleaner output
    const lastMessage = messages[messages.length - 1];
    console.log(`🤖 Assistant: ${lastMessage.content}\n`);
  }

  console.log("\n================================================");
  console.log("Multi-turn conversation with context complete!");
  console.log("================================================\n");
}

// Run the main function
main().catch(console.error);
