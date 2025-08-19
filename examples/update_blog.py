#!/usr/bin/env python3
"""
Example script for updating a 9takes personality blog

This demonstrates how to use the blog update workflow with different configurations.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_deep_research.updatePeopleBlogsEnhanced import update_blog, WorkflowStatus


async def update_single_blog():
    """Example: Update a single blog with default settings"""
    
    print("🚀 Starting blog update workflow...")
    print("-" * 50)
    
    # You can use a URL or Supabase ID/slug
    blog_identifier = "https://9takes.com/personality-analysis/Taylor-Swift"
    
    # Or use a Supabase slug if you have it configured
    # blog_identifier = "taylor-swift-enneagram"
    
    try:
        result = await update_blog(blog_identifier, {})
        
        if result['workflow_status'] == WorkflowStatus.COMPLETED:
            print("\n✅ Blog successfully updated!")
            print(f"   Person: {result.get('person', 'Unknown')}")
            print(f"   Enneagram Type: {result.get('enneagram_type', 'Unknown')}")
            print(f"   Research queries used: {len(result.get('search_queries', []))}")
            print(f"   Search results found: {len(result.get('search_results', []))}")
            
            if result.get('export_markdown'):
                print(f"\n📄 Markdown exported to: exports/")
            
            if result.get('save_to_supabase'):
                print(f"💾 Blog saved to Supabase")
                
        elif result['workflow_status'] == WorkflowStatus.ERROR:
            print(f"\n❌ Error occurred: {result.get('error_message', 'Unknown error')}")
        else:
            print(f"\n⚠️ Workflow ended with status: {result['workflow_status']}")
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")
        import traceback
        traceback.print_exc()


async def update_with_custom_config():
    """Example: Update a blog with custom configuration"""
    
    print("🎨 Starting blog update with custom configuration...")
    print("-" * 50)
    
    blog_identifier = "https://9takes.com/personality-analysis/Margot-Robbie"
    
    # Custom configuration
    config = {
        "search_method": "tavily",       # Options: tavily, perplexity, mcp_puppeteer
        "require_approval": True,        # Set to False to auto-approve
        "save_to_supabase": False,       # Just export, don't save to DB
        "export_markdown": True,         # Export markdown file
    }
    
    try:
        result = await update_blog(blog_identifier, config)
        
        if result['workflow_status'] == WorkflowStatus.COMPLETED:
            print("\n✅ Blog successfully updated with custom config!")
            print(f"   Search method used: {result.get('search_method', 'Unknown')}")
            print(f"   Approval required: {result.require_approval}")
            
    except Exception as e:
        print(f"\n❌ Exception occurred: {str(e)}")


async def batch_update_blogs():
    """Example: Update multiple blogs in sequence"""
    
    print("📚 Starting batch blog update...")
    print("-" * 50)
    
    blog_urls = [
        "https://9takes.com/personality-analysis/Taylor-Swift",
        "https://9takes.com/personality-analysis/Margot-Robbie",
        # Add more URLs here
    ]
    
    results = []
    for i, url in enumerate(blog_urls, 1):
        print(f"\n[{i}/{len(blog_urls)}] Updating: {url}")
        
        try:
            result = await update_blog(url, {})
            results.append({
                "url": url,
                "status": result.get('workflow_status'),
                "person": result.get('person'),
                "error": result.get('error_message')
            })
            
            if result['workflow_status'] == WorkflowStatus.COMPLETED:
                print(f"   ✅ Success: {result.get('person', 'Unknown')}")
            else:
                print(f"   ❌ Failed: {result.get('error_message', 'Unknown error')}")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            results.append({
                "url": url,
                "status": "ERROR",
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 50)
    print("BATCH UPDATE SUMMARY")
    print("=" * 50)
    
    successful = sum(1 for r in results if r.get("status") == WorkflowStatus.COMPLETED)
    print(f"✅ Successful: {successful}/{len(blog_urls)}")
    
    for r in results:
        status_icon = "✅" if r["status"] == WorkflowStatus.COMPLETED else "❌"
        print(f"{status_icon} {r.get('person', 'Unknown')} - {r['status']}")


def main():
    """Main entry point with menu"""
    
    print("\n" + "=" * 50)
    print("9TAKES BLOG UPDATE SYSTEM")
    print("=" * 50)
    
    print("\nSelect an option:")
    print("1. Update a single blog (default settings)")
    print("2. Update a blog with custom configuration")
    print("3. Batch update multiple blogs")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        asyncio.run(update_single_blog())
    elif choice == "2":
        asyncio.run(update_with_custom_config())
    elif choice == "3":
        asyncio.run(batch_update_blogs())
    elif choice == "4":
        print("Goodbye!")
        sys.exit(0)
    else:
        print("Invalid choice. Please run again.")
        sys.exit(1)


if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists(".env"):
        print("⚠️  Warning: .env file not found!")
        print("Please copy .env.example to .env and add your API keys.")
        print("\nRun: cp .env.example .env")
        sys.exit(1)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required API keys
    required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key) or os.getenv(key) in ["xxx", "sk-xxx"]]
    
    if missing_keys:
        print("❌ Missing required API keys in .env:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease edit .env and add your API keys.")
        sys.exit(1)
    
    # Run the main menu
    main()