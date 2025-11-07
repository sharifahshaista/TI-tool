# LLM Provider Setup Guide

This guide will help you configure your preferred LLM provider for the TI-Tool.

## Quick Comparison

| Provider | Cost | Speed | Privacy | Best For |
|----------|------|-------|---------|----------|
| **Azure OpenAI** | $$$ | Fast | Cloud | Enterprise, production |
| **OpenAI** | $$ | Fast | Cloud | Quick start, prototyping |
| **LM Studio** | Free | Medium | Local | Privacy-sensitive, offline work |

---

## 1. Azure OpenAI Setup

### Prerequisites
- Azure subscription
- Azure OpenAI resource created
- Model deployment created (e.g., GPT-4)

### Configuration Steps

1. Get your Azure OpenAI credentials from Azure Portal:
   - Navigate to your Azure OpenAI resource
   - Go to "Keys and Endpoint"
   - Copy: API Key, Endpoint, and Deployment Name

2. Edit your `.env` file:
   ```bash
   LLM_PROVIDER=azure
   AZURE_OPENAI_API_KEY=your_api_key_here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   OPENAI_API_VERSION=2023-12-01-preview
   AZURE_OPENAI_MODEL_NAME=your-deployment-name
   ```

3. Verify configuration:
   ```bash
   # The app will show "Azure OpenAI" in the LLM Provider selector
   ```

### Recommended Models
- **GPT-4**: Best quality, higher cost
- **GPT-4o**: Balanced performance and cost
- **GPT-4o-mini**: Fast and cost-effective
- **GPT-3.5-turbo**: Fastest, lowest cost

---

## 2. OpenAI API Setup

### Prerequisites
- OpenAI account ([sign up here](https://platform.openai.com/signup))
- API key with billing enabled

### Configuration Steps

1. Get your OpenAI API key:
   - Go to https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Copy the key (save it securely!)

2. Edit your `.env` file:
   ```bash
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your_api_key_here
   OPENAI_MODEL_NAME=gpt-4
   ```

3. Verify configuration:
   ```bash
   # The app will show "OpenAI" in the LLM Provider selector
   ```

### Recommended Models
- **gpt-4-turbo**: Latest GPT-4, faster and cheaper
- **gpt-4**: Standard GPT-4
- **gpt-4o**: Optimized for cost/performance
- **gpt-4o-mini**: Cost-effective option
- **gpt-3.5-turbo**: Fastest and cheapest

### Cost Estimates
- GPT-4: ~$0.03/1K tokens (input), ~$0.06/1K tokens (output)
- GPT-4-turbo: ~$0.01/1K tokens (input), ~$0.03/1K tokens (output)
- GPT-3.5-turbo: ~$0.0005/1K tokens (input), ~$0.0015/1K tokens (output)

---

## 3. LM Studio Setup (Local LLM)

### Prerequisites
- Computer with at least 8GB RAM (16GB+ recommended)
- ~5-10GB free disk space for models

### Installation Steps

1. **Download LM Studio:**
   - Visit https://lmstudio.ai/
   - Download for your OS (Windows, Mac, Linux)
   - Install the application

2. **Download a Model:**
   - Open LM Studio
   - Go to "Search" tab
   - Recommended models:
     - **Llama 3.1 8B** (8GB RAM minimum)
     - **Mistral 7B** (7GB RAM minimum)
     - **Phi-3 Medium** (Smaller, faster)
   - Click download and wait for completion

3. **Start the Local Server:**
   - Go to "Server" tab in LM Studio
   - Select your downloaded model
   - Click "Start Server"
   - Note the server URL (usually http://127.0.0.1:1234)

4. **Configure .env file:**
   ```bash
   LLM_PROVIDER=lm_studio
   LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
   LM_STUDIO_API_KEY=lm-studio
   ```

5. **Verify Configuration:**
   - Start the TI-Tool application
   - Check that "LM Studio (Local)" appears in the LLM Provider selector
   - The sidebar will show the server URL

### Recommended Models by Use Case

#### Best Overall (8B models)
- **Llama 3.1 8B Instruct** - Excellent balance
- **Mistral 7B Instruct v0.2** - Fast and capable

#### For Lower RAM Systems (3-4B models)
- **Phi-3 Mini** - Surprisingly capable for its size
- **Gemma 2 2B** - Google's efficient model

#### For High-End Systems (13B+ models)
- **Llama 3.1 70B** (Requires 40GB+ RAM)
- **Mixtral 8x7B** (Mixture of Experts, efficient)

### Performance Tips
- **GPU Acceleration**: Enable if you have a compatible NVIDIA/AMD GPU
- **Context Length**: Start with 2048, increase if needed
- **Temperature**: Use 0.7 for balanced outputs
- **Keep the server running**: Don't close LM Studio while using TI-Tool

---

## Switching Between Providers

You can easily switch providers in two ways:

### Method 1: Via UI (Recommended)
1. Open the TI-Tool Streamlit app
2. Look at the sidebar → Settings section
3. Select your preferred provider from the dropdown
4. The app will automatically use the new provider

### Method 2: Via .env File
1. Edit the `.env` file
2. Change the `LLM_PROVIDER` value:
   ```bash
   LLM_PROVIDER=azure    # or openai, or lm_studio
   ```
3. Restart the application

---

## Troubleshooting

### Azure OpenAI Issues

**Error: "Missing required environment variable"**
- Solution: Verify all three variables are set: API_KEY, ENDPOINT, API_VERSION

**Error: "Resource not found"**
- Solution: Check your ENDPOINT URL is correct and includes https://

**Error: "Deployment not found"**
- Solution: Verify MODEL_NAME matches your Azure deployment name exactly

### OpenAI Issues

**Error: "Incorrect API key"**
- Solution: Regenerate your API key from OpenAI dashboard

**Error: "Rate limit exceeded"**
- Solution: Add payment method or upgrade your OpenAI plan

**Error: "Model not found"**
- Solution: Check if the model name is spelled correctly

### LM Studio Issues

**Error: "Connection refused"**
- Solution: Ensure LM Studio server is started and running
- Check that the BASE_URL matches the LM Studio server URL

**Error: "Model not loaded"**
- Solution: Load a model in LM Studio before starting the server

**Slow Performance**
- Solution: Try a smaller model (3B-7B range)
- Enable GPU acceleration if available
- Reduce context length in LM Studio settings

---

## Best Practices

### For Production Use
- Use Azure OpenAI for enterprise reliability
- Set up proper monitoring and logging
- Implement rate limiting and retry logic

### For Development
- Use OpenAI for quick prototyping
- Use LM Studio for offline development
- Switch providers based on task complexity

### For Cost Optimization
- Use GPT-3.5-turbo or gpt-4o-mini for simple tasks
- Use GPT-4 only for complex reasoning
- Consider LM Studio for high-volume testing

### For Privacy-Sensitive Work
- Use LM Studio exclusively
- All processing happens locally
- No data sent to external services

---

## Getting Help

If you encounter issues:

1. Check the application logs for detailed error messages
2. Verify your .env file configuration
3. Test your API keys/connectivity outside the app
4. Refer to provider documentation:
   - [Azure OpenAI Docs](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
   - [OpenAI API Docs](https://platform.openai.com/docs)
   - [LM Studio Docs](https://lmstudio.ai/docs)

---

## Example .env Files

### Complete .env for Azure
```bash
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=abc123...
AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
OPENAI_API_VERSION=2023-12-01-preview
AZURE_OPENAI_MODEL_NAME=gpt-4
SEARXNG_URL=http://localhost:8888
```

### Complete .env for OpenAI
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-abc123...
OPENAI_MODEL_NAME=gpt-4-turbo
SEARXNG_URL=http://localhost:8888
```

### Complete .env for LM Studio
```bash
LLM_PROVIDER=lm_studio
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=lm-studio
SEARXNG_URL=http://localhost:8888
```
