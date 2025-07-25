import os
import gradio as gr
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent
import asyncio

# Load environment variables
load_dotenv()

# Set the API key in environment
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

class StockPriceResult(BaseModel):
    symbol: str
    price: float
    currency: str = "USD"
    message: str

# Create the agent globally with updated parameter
stock_agent = Agent(
    "groq:deepseek-r1-distill-llama-70b",
    output_type=StockPriceResult,  # Changed from result_type to output_type
    system_prompt="You are a helpful financial assistant that can look up stock prices. Use the get_stock_price function to look up current stock prices.",
)

@stock_agent.tool_plain
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a given symbol."""
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.fast_info.last_price
        if price is None:
            raise ValueError(f"Could not get price for {symbol}")
        return {"price": round(price, 2), "currency": "USD"}
    except Exception as e:
        raise ValueError(f"Error getting stock price: {str(e)}")

async def async_get_stock_info(query: str) -> str:
    try:
        # Run the query asynchronously
        result = await stock_agent.run(query)
        
        # Format the response
        response = f"Stock: {result.data.symbol}\n"
        response += f"Price: ${result.data.price:.2f} {result.data.currency}\n"
        response += f"{result.data.message}"
        return response
    except Exception as e:
        return f"Error: {str(e)}"

def get_stock_info(query: str) -> str:
    # Create a new event loop for the synchronous wrapper
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(async_get_stock_info(query))
        return result
    finally:
        loop.close()

# Create the Gradio interface
demo = gr.Interface(
    fn=get_stock_info,
    inputs=gr.Textbox(
        label="Ask about any stock price",
        placeholder="Example: What is the price of Tesla?",
    ),
    outputs=gr.Textbox(label="Stock Information"),
    title="Stock Price AI Assistant",
    description="Ask me about any stock price and I will help you find it!",
    examples=[
        ["What is Apple's current stock price?"],
        ["What is the price of Tesla stock?"],
        ["How much does Microsoft stock cost?"],
    ],
)

if __name__ == "__main__":
    demo.launch()