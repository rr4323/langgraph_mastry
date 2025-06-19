"""
LangGraph Intermediate: Advanced Tools
====================================

This script demonstrates how to implement more complex tools and tool patterns
in LangGraph using Google's Generative AI model.
"""

import os
import sys
import json
import time
import random
import requests
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define Pydantic models for structured tool inputs and outputs
class LocationInfo(BaseModel):
    """Information about a location."""
    city: str = Field(description="The name of the city")
    country: str = Field(description="The country where the city is located")
    
    def __str__(self):
        return f"{self.city}, {self.country}"

class WeatherInfo(BaseModel):
    """Weather information for a location."""
    location: LocationInfo = Field(description="The location information")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, cloudy, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage if available")
    
    def __str__(self):
        """String representation of the weather information."""
        humidity_str = f", humidity: {self.humidity}%" if self.humidity is not None else ""
        return f"Weather in {self.location}: {self.temperature}°C, {self.conditions}{humidity_str}"

class StockInfo(BaseModel):
    """Stock information."""
    symbol: str = Field(description="The stock symbol")
    price: float = Field(description="The current stock price")
    change: float = Field(description="The price change")
    change_percent: float = Field(description="The percentage price change")
    
    def __str__(self):
        """String representation of the stock information."""
        change_sign = "+" if self.change >= 0 else ""
        return f"{self.symbol}: ${self.price:.2f} ({change_sign}{self.change:.2f}, {change_sign}{self.change_percent:.2f}%)"

class NewsArticle(BaseModel):
    """News article information."""
    title: str = Field(description="The title of the article")
    source: str = Field(description="The source of the article")
    date: str = Field(description="The publication date")
    summary: str = Field(description="A summary of the article")
    url: Optional[str] = Field(None, description="The URL of the article")
    
    def __str__(self):
        """String representation of the news article."""
        return f"{self.title}\nSource: {self.source} | Date: {self.date}\n{self.summary}"

# Define advanced tools using the @tool decorator

@tool
def get_weather(location: str) -> WeatherInfo:
    """
    Get the current weather for a location.
    
    Args:
        location: The city and country (e.g., "London, UK")
    
    Returns:
        WeatherInfo: Weather information for the location
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Parse the location
    parts = location.split(",")
    city = parts[0].strip()
    country = parts[1].strip() if len(parts) > 1 else "Unknown"
    agent
    # Simulate weather data based on the location
    # In a real application, you would call a weather API
    weather_conditions = ["sunny", "cloudy", "rainy", "snowy", "windy"]
    temperatures = range(0, 35)
    humidity_values = range(30, 95)
    
    # Simple hash function to make the weather consistent for the same location
    import hashlib
    hash_value = int(hashlib.md5(location.encode()).hexdigest(), 16)
    weather_index = hash_value % len(weather_conditions)
    temp_index = hash_value % len(temperatures)
    humidity_index = hash_value % len(humidity_values)
    
    # Create and return the WeatherInfo object
    return WeatherInfo(
        location=LocationInfo(city=city, country=country),
        temperature=float(temperatures[temp_index]),
        conditions=weather_conditions[weather_index],
        humidity=float(humidity_values[humidity_index])
    )

@tool
def get_stock_price(symbol: str) -> StockInfo:
    """
    Get the current stock price for a given symbol.
    
    Args:
        symbol: The stock symbol (e.g., "AAPL" for Apple)
    
    Returns:
        StockInfo: Stock information
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Normalize the symbol
    symbol = symbol.upper().strip()
    
    # Simulate stock data based on the symbol
    # In a real application, you would call a stock API
    base_prices = {
        "AAPL": 180.0,
        "MSFT": 420.0,
        "GOOG": 150.0,
        "AMZN": 180.0,
        "META": 480.0,
        "TSLA": 240.0,
        "NVDA": 950.0,
        "NFLX": 620.0,
    }
    
    # Use a default price for unknown symbols
    base_price = base_prices.get(symbol, 100.0)
    
    # Add some randomness to the price
    random.seed(symbol)
    price = base_price * (1 + (random.random() - 0.5) * 0.1)
    change = price - base_price
    change_percent = (change / base_price) * 100
    
    # Create and return the StockInfo object
    return StockInfo(
        symbol=symbol,
        price=price,
        change=change,
        change_percent=change_percent
    )

@tool
def search_news(query: str, max_results: int = 3) -> List[NewsArticle]:
    """
    Search for news articles based on a query.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        List[NewsArticle]: A list of news articles
    """
    # Simulate API call delay
    time.sleep(1.5)
    
    # Simulate news data based on the query
    # In a real application, you would call a news API
    
    # Generate some fake news sources
    sources = ["The Daily News", "Tech Today", "Business Insider", "World Report", "Science Journal"]
    
    # Generate some fake dates (within the last week)
    current_time = time.time()
    dates = [time.strftime("%Y-%m-%d", time.localtime(current_time - random.randint(0, 7) * 86400)) for _ in range(10)]
    
    # Generate fake news articles based on the query
    articles = []
    for i in range(max_results):
        # Use a hash of the query and index to generate consistent results
        hash_value = hash(f"{query}_{i}")
        random.seed(hash_value)
        
        source = sources[random.randint(0, len(sources) - 1)]
        date = dates[random.randint(0, len(dates) - 1)]
        
        # Generate a title based on the query
        words = query.split()
        title_words = []
        for word in words:
            title_words.append(word.capitalize())
        
        title_prefixes = ["New Research on", "Breaking:", "The Future of", "Understanding", "Exploring"]
        title_suffixes = ["Revealed", "Explained", "in 2025", "and Its Impact", "- A Deep Dive"]
        
        prefix = title_prefixes[hash_value % len(title_prefixes)]
        suffix = title_suffixes[(hash_value // 10) % len(title_suffixes)]
        
        title = f"{prefix} {' '.join(title_words)} {suffix}"
        
        # Generate a summary based on the query
        summary = f"This article discusses {query} and its implications. "
        summary += f"Experts from {source} provide insights on the latest developments. "
        summary += f"Published on {date}, this article covers key aspects and future trends."
        
        # Create a NewsArticle object
        article = NewsArticle(
            title=title,
            source=source,
            date=date,
            summary=summary,
            url=f"https://example.com/news/{hash_value}"
        )
        
        articles.append(article)
    
    return articles

@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of a given text.
    
    Args:
        text: The text to analyze
    
    Returns:
        Dict: Sentiment analysis results
    """
    # Simulate API call delay
    time.sleep(1)
    
    # Simulate sentiment analysis
    # In a real application, you would call a sentiment analysis API
    
    # Simple rule-based sentiment analysis
    positive_words = ["good", "great", "excellent", "positive", "happy", "love", "best", "amazing"]
    negative_words = ["bad", "terrible", "negative", "sad", "hate", "worst", "awful", "poor"]
    
    # Count positive and negative words
    positive_count = sum(1 for word in text.lower().split() if word in positive_words)
    negative_count = sum(1 for word in text.lower().split() if word in negative_words)
    
    # Calculate sentiment score (-1 to 1)
    total_count = positive_count + negative_count
    if total_count == 0:
        sentiment_score = 0
    else:
        sentiment_score = (positive_count - negative_count) / total_count
    
    # Determine sentiment label
    if sentiment_score > 0.3:
        sentiment = "positive"
    elif sentiment_score < -0.3:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    
    # Return sentiment analysis results
    return {
        "sentiment": sentiment,
        "score": sentiment_score,
        "positive_words": positive_count,
        "negative_words": negative_count,
        "confidence": abs(sentiment_score) * 0.7 + 0.3  # Scale to 0.3-1.0
    }

# Tool composition: a tool that uses other tools
@tool
def get_market_summary(symbols: List[str]) -> Dict[str, Any]:
    """
    Get a summary of market information for a list of stock symbols.
    
    Args:
        symbols: List of stock symbols (e.g., ["AAPL", "MSFT", "GOOG"])
    
    Returns:
        Dict: Market summary information
    """
    # Get stock information for each symbol
    stocks = [get_stock_price(symbol) for symbol in symbols]
    
    # Calculate market statistics
    total_value = sum(stock.price for stock in stocks)
    avg_price = total_value / len(stocks)
    avg_change_percent = sum(stock.change_percent for stock in stocks) / len(stocks)
    
    # Determine market trend
    if avg_change_percent > 1.0:
        trend = "strong bullish"
    elif avg_change_percent > 0.2:
        trend = "bullish"
    elif avg_change_percent > -0.2:
        trend = "neutral"
    elif avg_change_percent > -1.0:
        trend = "bearish"
    else:
        trend = "strong bearish"
    
    # Get related news
    news_query = f"stock market {trend}"
    news = search_news(news_query, max_results=2)
    
    # Return market summary
    return {
        "stocks": stocks,
        "statistics": {
            "total_value": total_value,
            "average_price": avg_price,
            "average_change_percent": avg_change_percent,
            "trend": trend
        },
        "news": news
    }

def create_agent_with_advanced_tools():
    """Create a LangGraph agent with advanced tools."""
    print("Creating a LangGraph agent with advanced tools...")
    
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Define the tools
    tools = [
        get_weather,
        get_stock_price,
        search_news,
        analyze_sentiment,
        get_market_summary
    ]
    
    # Create the agent with advanced tools
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt="""You are a financial and news assistant with access to advanced tools.
You can provide weather information, stock prices, news articles, sentiment analysis, and market summaries.
Use your tools appropriately to provide accurate and helpful information.
Always think step by step about which tool would be most appropriate to use.
"""
    )
    
    return agent

def interact_with_agent(agent):
    """Interact with the agent by sending messages and receiving responses."""
    print("\n" + "=" * 50)
    print("Interacting with your LangGraph Agent with Advanced Tools")
    print("=" * 50)
    print("Available tools:")
    print("- get_weather: Get weather information for a location")
    print("- get_stock_price: Get stock price information")
    print("- search_news: Search for news articles")
    print("- analyze_sentiment: Analyze sentiment of text")
    print("- get_market_summary: Get a summary of market information")
    print("(Type 'exit' to end the conversation)")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'exit':
            print("\nEnding conversation.")
            break
        
        # Invoke the agent with the message
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}
        )
        
        # Print the agent's response
        print(f"\nAgent: {response['messages'][-1].content}")

def demonstrate_tool_usage():
    """Demonstrate the usage of advanced tools."""
    print("\n" + "=" * 50)
    print("Demonstrating Advanced Tool Usage")
    print("=" * 50)
    
    # Demonstrate get_weather tool
    print("\n1. Using get_weather tool:")
    weather = get_weather("London, UK")
    print(f"Result: {weather}")
    
    # Demonstrate get_stock_price tool
    print("\n2. Using get_stock_price tool:")
    stock = get_stock_price("AAPL")
    print(f"Result: {stock}")
    
    # Demonstrate search_news tool
    print("\n3. Using search_news tool:")
    news = search_news("artificial intelligence")
    for i, article in enumerate(news, 1):
        print(f"\nArticle {i}:")
        print(article)
    
    # Demonstrate analyze_sentiment tool
    print("\n4. Using analyze_sentiment tool:")
    sentiment = analyze_sentiment("I really love this product, it's amazing and works great!")
    print(f"Result: {json.dumps(sentiment, indent=2)}")
    
    # Demonstrate get_market_summary tool (tool composition)
    print("\n5. Using get_market_summary tool (tool composition):")
    market_summary = get_market_summary(["AAPL", "MSFT", "GOOG"])
    print("Market Summary:")
    print(f"Stocks:")
    for stock in market_summary["stocks"]:
        print(f"  {stock}")
    print(f"Statistics:")
    for key, value in market_summary["statistics"].items():
        print(f"  {key}: {value}")
    print(f"Related News:")
    for i, article in enumerate(market_summary["news"], 1):
        print(f"\nArticle {i}:")
        print(article)

def main():
    """Main function to demonstrate advanced tools."""
    print("=" * 50)
    print("Advanced Tools in LangGraph")
    print("=" * 50)
    
    # Let the user choose a mode
    print("\nSelect a mode:")
    print("1. Interactive conversation with advanced tools")
    print("2. Automated demonstration of advanced tools")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-2): "))
            if choice == 1:
                agent = create_agent_with_advanced_tools()
                interact_with_agent(agent)
                break
            elif choice == 2:
                demonstrate_tool_usage()
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
