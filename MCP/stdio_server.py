from mcp.server.fastmcp import FastMCP

# Create MCP server
mcp = FastMCP("weather")

# Define tool
@mcp.tool()
def get_weather(city: str) -> str:
    return f"Weather in {city}: Sunny, 25°C"

# Run MCP server over stdio
if __name__ == "__main__":
    mcp.run(transport="stdio")