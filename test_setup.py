#!/usr/bin/env python3
"""
Test script to verify your setup is working correctly
Run this after setting up your environment and API keys
"""

import os
import sys
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"{Fore.GREEN}✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"{Fore.RED}❌ {package_name} is NOT installed")
        return False

def check_env_var(var_name, required=True):
    """Check if an environment variable is set"""
    value = os.getenv(var_name)
    if value and value != "xxx" and "your_" not in value and "your-" not in value:
        # Mask the actual value for security
        masked = value[:4] + "..." + value[-4:] if len(value) > 8 else "***"
        print(f"{Fore.GREEN}✅ {var_name} is set ({masked})")
        return True
    else:
        if required:
            print(f"{Fore.RED}❌ {var_name} is NOT set or using default value")
        else:
            print(f"{Fore.YELLOW}⚠️  {var_name} is optional and not set")
        return False

def main():
    print(f"{Fore.CYAN}{'='*50}")
    print(f"{Fore.CYAN}Open Deep Research - Setup Verification")
    print(f"{Fore.CYAN}{'='*50}\n")
    
    all_good = True
    
    # Check Python version
    print(f"{Fore.BLUE}1. Python Version Check:")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 9:
        print(f"{Fore.GREEN}✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} (Compatible)")
    else:
        print(f"{Fore.RED}❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (Requires 3.9+)")
        all_good = False
    
    # Check required packages
    print(f"\n{Fore.BLUE}2. Required Packages:")
    packages = [
        ("langchain", "langchain"),
        ("langchain_openai", "langchain-openai"),
        ("langchain_anthropic", "langchain-anthropic"),
        ("langgraph", "langgraph"),
        ("tavily", "tavily-python"),
        ("pydantic", "pydantic"),
    ]
    
    for module, package in packages:
        if not check_import(module, package):
            all_good = False
    
    # Check optional packages
    print(f"\n{Fore.BLUE}3. Optional Packages:")
    optional = [
        ("supabase", "supabase"),
        ("langchain_groq", "langchain-groq"),
    ]
    
    for module, package in optional:
        check_import(module, package)
    
    # Load .env file if it exists
    if os.path.exists(".env"):
        print(f"\n{Fore.BLUE}4. Loading .env file...")
        from dotenv import load_dotenv
        load_dotenv()
        print(f"{Fore.GREEN}✅ .env file loaded")
    else:
        print(f"\n{Fore.YELLOW}⚠️  No .env file found. Using environment variables.")
    
    # Check required environment variables
    print(f"\n{Fore.BLUE}5. Required API Keys:")
    required_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
    ]
    
    for var in required_vars:
        if not check_env_var(var, required=True):
            all_good = False
    
    # Check Supabase variables (required for blog updates)
    print(f"\n{Fore.BLUE}6. Supabase Configuration (Required for blog updates):")
    supabase_vars = [
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_KEY",
    ]
    
    supabase_configured = True
    for var in supabase_vars:
        if not check_env_var(var, required=False):
            supabase_configured = False
    
    # Check optional environment variables
    print(f"\n{Fore.BLUE}7. Optional API Keys:")
    optional_vars = [
        "GROQ_API_KEY",
        "PERPLEXITY_API_KEY",
        "LANGCHAIN_API_KEY",
    ]
    
    for var in optional_vars:
        check_env_var(var, required=False)
    
    # Summary
    print(f"\n{Fore.CYAN}{'='*50}")
    if all_good and supabase_configured:
        print(f"{Fore.GREEN}🎉 Your setup is complete and ready to use!")
        print(f"\n{Fore.CYAN}You can now run:")
        print(f"{Fore.WHITE}python examples/update_blog.py")
    elif all_good:
        print(f"{Fore.YELLOW}⚠️  Core setup is complete but Supabase is not configured.")
        print(f"{Fore.YELLOW}You can fetch blogs from URLs but not from Supabase.")
    else:
        print(f"{Fore.RED}❌ Setup is incomplete. Please:")
        print(f"{Fore.WHITE}1. Run: ./setup.sh")
        print(f"{Fore.WHITE}2. Edit .env file with your API keys")
        print(f"{Fore.WHITE}3. Run this test again")
    print(f"{Fore.CYAN}{'='*50}")

if __name__ == "__main__":
    # Try to import colorama, install if not available
    try:
        from colorama import Fore, Style, init
    except ImportError:
        print("Installing colorama for colored output...")
        os.system(f"{sys.executable} -m pip install colorama")
        from colorama import Fore, Style, init
    
    # Try to import dotenv
    try:
        from dotenv import load_dotenv
    except ImportError:
        print("Installing python-dotenv for environment variables...")
        os.system(f"{sys.executable} -m pip install python-dotenv")
        from dotenv import load_dotenv
    
    main()