import weave
import os
import time
import requests


class Gpt4oMiniModel(weave.Model):
    temperature: float = 0.2

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        # Create an OpenAI client for OpenRouter
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"]
        )

        # Call the GPT-4o-mini model
        completion = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.temperature
        )

        return completion.choices[0].message.content
        
        
class LlamaModel(weave.Model):
    temperature: float = 0.3
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        data = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }
        
        response = self.make_openrouter_request(data)
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        

class LlamaLargeModel(weave.Model):
    temperature: float = 0.3
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        data = {
            "model": "meta-llama/llama-3.1-405b-instruct:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }
        
        response = self.make_openrouter_request(data)
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")


class Phi3Model(weave.Model):
    temperature: float = 0.2
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        data = {
            "model": "microsoft/phi-3-medium-128k-instruct:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }
        
        response = self.make_openrouter_request(data)
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")


class OpenChatModel(weave.Model):
    temperature: float = 0.2
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        data = {
            "model": "openchat/openchat-7b:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }
        
        response = self.make_openrouter_request(data)
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")


class MistralModel(weave.Model):
    temperature: float = 0.2
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        data = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }
        
        response = self.make_openrouter_request(data)
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")


class QwenModel(weave.Model):
    temperature: float = 0.2
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        # Construct the data payload for the OpenRouter API
        data = {
            "model": "qwen/qwen-2-7b-instruct:free",  # Specify the Mistral model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }

        # Make the OpenRouter API request
        response = self.make_openrouter_request(data)
        
        # Return the content of the response
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            # Make the request to the OpenRouter API endpoint
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            # Check for errors in the response
            response.raise_for_status()
            
            # Return the JSON response
            return response.json()

        except requests.RequestException as e:
            # Raise a custom error if the request fails
            raise Exception(f"API request failed: {str(e)}")
        
class GeminiModel(weave.Model):
    temperature: float = 0.2
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        # Construct the data payload for the OpenRouter API
        data = {
            "model": "google/gemini-exp-1206:free",  # Specify the Gemini model
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }

        # Make the OpenRouter API request
        response = self.make_openrouter_request(data)
        
        # Return the content of the response
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            # Make the request to the OpenRouter API endpoint
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json",
                },
                data=data
            )
            # Check for errors in the response
            response.raise_for_status()
            
            # Return the JSON response
            return response.json()

        except requests.RequestException as e:
            # Raise a custom error if the request fails
            raise Exception(f"API request failed: {str(e)}")
        

class GemmaModel(weave.Model):
    temperature: float = 0.3
    provider: str

    @weave.op()
    def predict(self, prompt: str) -> str:
        time.sleep(5)  # Delay of 5s
        data = {
            "model": "google/gemma-2-9b-it:free",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "provider": {
                "order": [self.provider],
                "allow_fallbacks": False
            }
        }
        
        response = self.make_openrouter_request(data)
        return response['choices'][0]['message']['content']

    @weave.op()
    def make_openrouter_request(self, data):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers = {
                    "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=data
            )
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")