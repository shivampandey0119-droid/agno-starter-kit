from mcp.server.fastmcp import FastMCP

# Create MCP server for math operations
mcp = FastMCP("math-server")

@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

if __name__ == "__main__":
    # Use 'http' transport (recommended over standalone 'sse')
    # Streamable HTTP is the modern standard for web-native MCP servers.
    mcp.settings.port = 8002
    mcp.run(transport="streamable-http")
