from mcp.server.fastmcp import FastMCP

# Create server with explicit uppercase log_level
mcp = FastMCP("Echo Server", log_level="INFO", port=8080)


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def get_secret_word() -> str:
    return "Strawberry"


@mcp.tool()
def get_current_weather(city: str) -> str:
    # Return mocked weather data to avoid external dependencies in tests
    return f"Weather report: {city}\nTemperature: 22°C (72°F)\nCondition: Partly cloudy\nHumidity: 65%\nWind: 10 km/h"


if __name__ == "__main__":
    mcp.run(transport="sse")
