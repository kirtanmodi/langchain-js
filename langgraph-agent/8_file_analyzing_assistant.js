// 8_file_analyzing_assistant.js
// A multi-step file analyzing assistant that can process documents and provide insights

import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage, SystemMessage, AIMessage } from "@langchain/core/messages";
import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { Document } from "@langchain/core/documents";
import * as fs from "fs/promises";
import * as path from "path";

// Define states for our file analyzer
const ANALYZER_STATES = {
  INITIALIZE: "initialize",
  UNDERSTAND: "understand",
  GENERATE_INSIGHTS: "generate_insights",
  ANSWER_QUESTIONS: "answer_questions",
  DONE: "done",
};

// Initialize the language model for different tasks
const llm = new ChatOpenAI({
  model: "gpt-4o-mini",
  temperature: 0.2,
});

// Text splitter to break large documents into manageable chunks
const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 2000,
  chunkOverlap: 200,
});

/**
 * File Analyzer class that orchestrates the analysis process
 */
class FileAnalyzer {
  constructor() {
    this.graph = null;
    this.fileContent = null;
    this.filePath = null;
    this.fileName = null;
    this.fileType = null;
    this.chunks = [];
    this.summary = null;
    this.insights = null;
  }

  async initialize() {
    // Create and compile the state graph
    this.graph = this.createStateGraph();
    return this;
  }

  // Create the analysis workflow as a state graph
  createStateGraph() {
    // Define all the nodes in our analysis workflow
    const workflow = new StateGraph(MessagesAnnotation)
      .addNode(ANALYZER_STATES.INITIALIZE, this.initializeAnalysis.bind(this))
      .addNode(ANALYZER_STATES.UNDERSTAND, this.understandContent.bind(this))
      .addNode(ANALYZER_STATES.GENERATE_INSIGHTS, this.generateInsights.bind(this))
      .addNode(ANALYZER_STATES.ANSWER_QUESTIONS, this.answerQuestions.bind(this))

      // Add conditional routing between states
      .addEdge("__start__", ANALYZER_STATES.INITIALIZE)
      .addConditionalEdges(ANALYZER_STATES.INITIALIZE, this.routeAfterInitialize.bind(this), {
        [ANALYZER_STATES.UNDERSTAND]: ANALYZER_STATES.UNDERSTAND,
        [ANALYZER_STATES.DONE]: "__end__",
      })
      .addEdge(ANALYZER_STATES.UNDERSTAND, ANALYZER_STATES.GENERATE_INSIGHTS)
      .addEdge(ANALYZER_STATES.GENERATE_INSIGHTS, ANALYZER_STATES.ANSWER_QUESTIONS)
      .addEdge(ANALYZER_STATES.ANSWER_QUESTIONS, "__end__");

    return workflow.compile();
  }

  // Load and prepare the file for analysis
  async initializeAnalysis({ messages }) {
    console.log("🔍 Initializing file analysis...");

    // Extract file path from the user message
    const userMessage = messages.filter((m) => m instanceof HumanMessage).pop();
    if (!userMessage) {
      return {
        messages: [new AIMessage("No file path provided. Please provide a valid file path to analyze.")],
      };
    }

    const requestText = userMessage.content;
    // Try to extract a file path from the request
    const filePath = this.extractFilePath(requestText);

    if (!filePath) {
      return {
        messages: [new AIMessage("Could not identify a file path in your message. Please provide a valid file path to analyze.")],
      };
    }

    try {
      // Load the file and get its content
      this.filePath = filePath;
      this.fileName = path.basename(filePath);
      this.fileType = path.extname(filePath).toLowerCase();

      // Read the file
      this.fileContent = await fs.readFile(filePath, "utf8");

      // Split the content into manageable chunks
      if (this.fileContent) {
        const docObj = new Document({
          pageContent: this.fileContent,
          metadata: {
            source: this.fileName,
            type: this.fileType,
          },
        });

        this.chunks = await textSplitter.splitDocuments([docObj]);
        console.log(`📄 File loaded and split into ${this.chunks.length} chunks`);

        return {
          messages: [
            ...messages,
            new AIMessage(
              `I've loaded "${this.fileName}" successfully. The file is ${(this.fileContent.length / 1024).toFixed(
                2
              )} KB in size and has been split into ${this.chunks.length} chunks for analysis. I'll now analyze the content.`
            ),
          ],
        };
      } else {
        return {
          messages: [...messages, new AIMessage(`The file "${filePath}" appears to be empty. Please provide a file with content to analyze.`)],
        };
      }
    } catch (error) {
      console.error("Error loading file:", error);
      return {
        messages: [...messages, new AIMessage(`Error loading the file: ${error.message}. Please check if the file exists and is accessible.`)],
      };
    }
  }

  // Extract potential file path from user message
  extractFilePath(text) {
    // Try to find patterns that look like file paths
    const pathRegex = /(["']?)([/\\]?[\w\s\-./\\]+\.\w+)\1/g;
    const matches = [...text.matchAll(pathRegex)];

    if (matches.length > 0) {
      // Return the most likely file path (usually the longest match)
      return matches.reduce((longest, match) => (match[2].length > longest.length ? match[2] : longest), "");
    }

    // Fallback: Try to find any word that ends with a common file extension
    const extensionRegex = /\b[\w\-./\\]+(\.js|\.ts|\.json|\.txt|\.md|\.csv|\.html|\.css|\.py|\.java|\.xml)\b/i;
    const extMatch = text.match(extensionRegex);

    return extMatch ? extMatch[0] : null;
  }

  // Route to either understanding or done state
  routeAfterInitialize({ messages }) {
    const lastMessage = messages[messages.length - 1];

    // If there was an error loading the file, we're done
    if (
      lastMessage.content.includes("Error loading") ||
      lastMessage.content.includes("Could not identify") ||
      lastMessage.content.includes("appears to be empty")
    ) {
      return ANALYZER_STATES.DONE;
    }

    // Otherwise, proceed to understanding the content
    return ANALYZER_STATES.UNDERSTAND;
  }

  // Analyze and understand the file content
  async understandContent({ messages }) {
    console.log("📚 Understanding file content...");

    // Create a summary from the chunks
    let contentForSummary = "";

    // Limit the amount of content to summarize to avoid token limits
    const maxChunks = Math.min(this.chunks.length, 10);
    for (let i = 0; i < maxChunks; i++) {
      contentForSummary += this.chunks[i].pageContent + "\n\n";
    }

    if (this.chunks.length > maxChunks) {
      contentForSummary += `[... and ${this.chunks.length - maxChunks} more chunks ...]`;
    }

    // Create a system prompt for the summary generation
    const summarySystemPrompt = new SystemMessage(`
      You are a file analysis assistant. You're analyzing a file named "${this.fileName}" of type "${this.fileType}".
      Your task is to provide a concise but comprehensive summary of the file content.
      Focus on the main purpose, structure, and key elements in the file.
      Keep your summary under 400 words.
    `);

    // Generate the summary
    const summaryResponse = await llm.invoke([summarySystemPrompt, new HumanMessage(`Here is the content to summarize:\n\n${contentForSummary}`)]);

    // Store the summary for later use
    this.summary = summaryResponse.content;

    return {
      messages: [...messages, new AIMessage(`I've analyzed the file. Here's a summary:\n\n${this.summary}`)],
    };
  }

  // Generate deeper insights about the file
  async generateInsights({ messages }) {
    console.log("💡 Generating insights...");

    // Use the summary and file metadata to generate insights
    const insightsSystemPrompt = new SystemMessage(`
      You are a file analysis assistant. You've already summarized the file "${this.fileName}" (${this.fileType}).
      Now, generate insightful observations about this file that would be valuable to the user.
      Consider the following aspects:
      - Code quality, patterns, and potential improvements (if it's code)
      - Structure and organization
      - Potential issues or vulnerabilities
      - Best practices followed or missing
      - Any unique or interesting aspects worth highlighting
      
      Be specific and helpful with your insights.
    `);

    // Sample a few random chunks to get more context
    let randomChunks = "";
    if (this.chunks.length > 0) {
      const sampleSize = Math.min(3, this.chunks.length);
      const indices = Array.from({ length: this.chunks.length }, (_, i) => i);

      // Shuffle array to get random indices
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      // Take the first few random indices
      for (let i = 0; i < sampleSize; i++) {
        randomChunks += `Chunk ${i + 1}:\n${this.chunks[indices[i]].pageContent}\n\n`;
      }
    }

    // Generate insights
    const insightsResponse = await llm.invoke([
      insightsSystemPrompt,
      new HumanMessage(`
        Here is the summary I created earlier:
        ${this.summary}
        
        And here are some additional content samples from the file:
        ${randomChunks}
        
        Based on this information, provide valuable insights about this file.
      `),
    ]);

    // Store the insights
    this.insights = insightsResponse.content;

    return {
      messages: [...messages, new AIMessage(`Based on my analysis, here are some insights about this file:\n\n${this.insights}`)],
    };
  }

  // Answer specific questions about the file
  async answerQuestions({ messages }) {
    console.log("❓ Ready to answer questions...");

    // Prepare the assistant to answer questions
    const finalMessage = new AIMessage(`
I've completed my analysis of "${this.fileName}". Here's what I found:

## Summary
${this.summary}

## Key Insights
${this.insights}

## Next Steps
You can ask me specific questions about this file, such as:
- Questions about specific parts or functions
- Requests for suggestions on improving the code
- Clarification on any aspect of the file
- How certain components work

What would you like to know about this file?
    `);

    return {
      messages: [...messages, finalMessage],
    };
  }

  // Main method to analyze a file
  async analyzeFile(filePath) {
    // Create the initial message with the file path
    const initialMessages = [new HumanMessage(`Please analyze this file: ${filePath}`)];

    // Run the analysis graph
    const result = await this.graph.invoke({
      messages: initialMessages,
    });

    return result;
  }

  // Method to ask follow-up questions about the analyzed file
  async askQuestion(question, previousMessages) {
    // Add the new question to the conversation
    const updatedMessages = [...previousMessages, new HumanMessage(question)];

    // Create a system prompt for answering questions
    const questionSystemPrompt = new SystemMessage(`
      You are a file analysis assistant. You've analyzed the file "${this.fileName}" (${this.fileType}).
      A user is asking a specific question about this file. Use your knowledge of the file to provide a detailed answer.
      If you don't know the answer with certainty, be honest about it.
    `);

    // Find relevant chunks to the question
    let relevantChunks = [];
    if (this.chunks.length > 0) {
      // Simple keyword matching to find relevant chunks
      const keywords = question
        .toLowerCase()
        .split(/\s+/)
        .filter((word) => word.length > 3) // Only consider words with > 3 characters
        .map((word) => word.replace(/[^\w]/g, "")); // Remove non-word characters

      if (keywords.length > 0) {
        relevantChunks = this.chunks
          .filter((chunk) => {
            const content = chunk.pageContent.toLowerCase();
            return keywords.some((keyword) => content.includes(keyword));
          })
          .slice(0, 3); // Limit to 3 chunks
      } else {
        // If no good keywords, just take first couple of chunks
        relevantChunks = this.chunks.slice(0, 2);
      }
    }

    // Combine relevant chunks
    let relevantContent = relevantChunks.map((chunk) => chunk.pageContent).join("\n\n");

    // Generate answer
    const answer = await llm.invoke([
      questionSystemPrompt,
      new HumanMessage(`
        Question: ${question}
        
        Here is relevant content from the file to help answer:
        ${relevantContent}
        
        Summary of file:
        ${this.summary}
        
        Please provide a detailed answer to the question.
      `),
    ]);

    // Add the answer to the conversation
    return [...updatedMessages, answer];
  }
}

// Format message content for display
function formatMessage(message) {
  if (message instanceof HumanMessage) {
    return `🧑 User: ${message.content}`;
  } else if (message instanceof AIMessage) {
    return `🤖 Assistant: ${message.content}`;
  }
  return "";
}

// Main function to demonstrate the file analyzer
async function main() {
  console.log("\n🔎 Starting File Analysis Assistant...\n");

  // Initialize the file analyzer
  const analyzer = await new FileAnalyzer().initialize();

  // Define a sample file to analyze
  const filePath = "./8_file_analyzing_assistant.js";

  try {
    // Analyze the file
    const result = await analyzer.analyzeFile(filePath);

    // check if error
    if (result.messages.some((m) => m.content.includes("Error loading"))) {
      console.error("Error loading file.");
      return;
    }

    // Display the analysis results
    console.log("\n=== File Analysis Results ===\n");
    for (const message of result.messages) {
      console.log(formatMessage(message));
      console.log();
    }

    // Simulate asking follow-up questions
    const questions = ["What are the main functions in this file?", "How could this code be improved?"];

    let currentMessages = result.messages;

    for (const question of questions) {
      console.log(`\n✋ User asking: ${question}\n`);

      // Get the answer
      currentMessages = await analyzer.askQuestion(question, currentMessages);

      // Display just the last message (the answer)
      const lastMessage = currentMessages[currentMessages.length - 1];
      console.log(`🤖 Assistant: ${lastMessage.content}\n`);
    }

    console.log("\n✅ File analysis demonstration complete.\n");
  } catch (error) {
    console.error("Error during file analysis:", error);
  }
}

// Run the demonstration
main().catch(console.error);
