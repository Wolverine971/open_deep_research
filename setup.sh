#!/bin/bash

echo "🚀 Setting up Open Deep Research Blog Update System"
echo "=================================================="

# Create virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "📚 Installing open_deep_research package..."
pip install -e .

# Install development dependencies
echo "🛠️ Installing development dependencies..."
pip install -e ".[dev]"

# Install Supabase for blog updates
echo "🗄️ Installing Supabase client..."
pip install supabase

# Create exports directory for markdown files
echo "📁 Creating exports directory..."
mkdir -p exports

# Copy .env.example if .env doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env and add your API keys!"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file and add your API keys:"
echo "   - OPENAI_API_KEY"
echo "   - ANTHROPIC_API_KEY"
echo "   - TAVILY_API_KEY"
echo "   - SUPABASE_URL"
echo "   - SUPABASE_ANON_KEY"
echo "   - SUPABASE_SERVICE_KEY"
echo ""
echo "2. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the blog update workflow:"
echo "   python -c \"from open_deep_research.updatePeopleBlogsEnhanced import update_blog; import asyncio; asyncio.run(update_blog('your-blog-url'))\""
echo ""
echo "Or use the Python script examples in the docs!"