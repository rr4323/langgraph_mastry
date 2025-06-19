"""
LangGraph Intermediate: Structured Output
=======================================

This script demonstrates how to work with structured outputs in LangGraph
using Google's Generative AI model and Pydantic models.
"""

import os
import sys
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv()

# Define Pydantic models for structured output
class WeatherInfo(BaseModel):
    """Weather information for a location."""
    location: str = Field(description="The name of the location")
    temperature: float = Field(description="The temperature in Celsius")
    conditions: str = Field(description="The weather conditions (e.g., sunny, cloudy, rainy)")
    humidity: Optional[float] = Field(None, description="The humidity percentage if available")
    
    def __str__(self):
        """String representation of the weather information."""
        humidity_str = f", humidity: {self.humidity}%" if self.humidity is not None else ""
        return f"Weather in {self.location}: {self.temperature}°C, {self.conditions}{humidity_str}"

class MovieRecommendation(BaseModel):
    """Movie recommendation based on user preferences."""
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The year the movie was released")
    genre: str = Field(description="The primary genre of the movie")
    director: str = Field(description="The director of the movie")
    reason: str = Field(description="The reason this movie is being recommended")
    
    def __str__(self):
        """String representation of the movie recommendation."""
        return f"{self.title} ({self.year}) - {self.genre}, directed by {self.director}\nRecommended because: {self.reason}"

class MovieRecommendations(BaseModel):
    """A list of movie recommendations."""
    recommendations: List[MovieRecommendation] = Field(description="List of movie recommendations")
    
    def __str__(self):
        """String representation of the movie recommendations."""
        return "\n\n".join([str(rec) for rec in self.recommendations])

# Define some tools for our agent
def get_weather(location: str) -> str:
    """Get the current weather for a location (simulated)."""
    # This is a simulated weather function
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
    
    return f"The weather in {location} is currently {weather_conditions[weather_index]} with a temperature of {temperatures[temp_index]}°C and humidity of {humidity_values[humidity_index]}%"

def create_agent_with_structured_output(output_model):
    """Create a LangGraph agent with structured output."""
    print(f"Creating a LangGraph agent with structured output using {output_model.__name__}...")
    
    # Initialize the chat model with Google's Gemini Pro
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        convert_system_message_to_human=True
    )
    
    # Define the tools
    tools = [get_weather]
    
    # Create a custom prompt for the structured output
    if output_model == WeatherInfo:
        prompt = """You are a weather assistant.
You provide accurate weather information for different locations.
When asked about the weather, use your tools to get the information.
"""
    elif output_model in (MovieRecommendation, MovieRecommendations):
        prompt = """You are a movie recommendation assistant.
You provide personalized movie recommendations based on user preferences.
Consider the user's stated preferences, genres, directors, or themes they enjoy.
Provide detailed recommendations with reasons why each movie would appeal to the user.
"""
    else:
        prompt = "You are a helpful assistant."
    
    # Create the agent with structured output
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        response_format=output_model,  # This enables structured output
        debug=True  # This enables debug mode
    )
    
    return agent

def demonstrate_weather_structured_output():
    """Demonstrate structured output with weather information."""
    print("\n" + "=" * 50)
    print("Demonstrating Structured Output: Weather Information")
    print("=" * 50)
    
    # Create an agent with WeatherInfo structured output
    agent = create_agent_with_structured_output(WeatherInfo)
    
    # Ask about the weather
    locations = ["London", "Tokyo", "New York", "Sydney"]
    
    for location in locations:
        print(f"\nAsking about weather in {location}...")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": f"What's the weather like in {location}?"}]}
        )
        
        # Access the structured response
        structured_response = response.get("structured_response")
        
        print(f"Raw structured response: {structured_response}")
        print(f"Formatted output: {structured_response}")
        
        # Access specific fields
        print(f"Location: {structured_response.location}")
        print(f"Temperature: {structured_response.temperature}°C")
        print(f"Conditions: {structured_response.conditions}")
        if structured_response.humidity is not None:
            print(f"Humidity: {structured_response.humidity}%")

def demonstrate_movie_structured_output():
    """Demonstrate structured output with movie recommendations."""
    print("\n" + "=" * 50)
    print("Demonstrating Structured Output: Movie Recommendations")
    print("=" * 50)
    
    # Create an agent with MovieRecommendations structured output
    agent = create_agent_with_structured_output(MovieRecommendations)
    
    # Ask for movie recommendations
    queries = [
        "Can you recommend some sci-fi movies with mind-bending plots?",
        "I enjoy romantic comedies from the 90s. What would you recommend?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        
        # Access the structured response
        structured_response = response.get("structured_response")
        
        print(f"Formatted recommendations:\n{structured_response}")
        
        # Access specific fields
        print(f"\nNumber of recommendations: {len(structured_response.recommendations)}")
        for i, rec in enumerate(structured_response.recommendations, 1):
            print(f"\nRecommendation {i}:")
            print(f"Title: {rec.title}")
            print(f"Year: {rec.year}")
            print(f"Genre: {rec.genre}")
            print(f"Director: {rec.director}")
            print(f"Reason: {rec.reason}")

def main():
    """Main function to demonstrate structured output."""
    print("=" * 50)
    print("Structured Output in LangGraph")
    print("=" * 50)
    
    # Let the user choose a demonstration
    print("\nSelect a demonstration:")
    print("1. Weather Information (single object output)")
    print("2. Movie Recommendations (list of objects output)")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-2): "))
            if choice == 1:
                demonstrate_weather_structured_output()
                break
            elif choice == 2:
                demonstrate_movie_structured_output()
                break
            else:
                print("Please enter 1 or 2.")
        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
