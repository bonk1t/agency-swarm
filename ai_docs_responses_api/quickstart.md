/
Playground
Dashboard
Docs
API reference

Developer quickstart
Take your first steps with the OpenAI API.
The OpenAI API provides a simple interface to state-of-the-art AI models for text generation, natural language processing, computer vision, and more. This example generates text output from a prompt, as you might using ChatGPT.

Generate text from a model
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
Data retention for model responses
Configure your development environment
Install and configure an official OpenAI SDK to run the code above.

Responses starter app
Start building with the Responses API

Text generation and prompting
Learn more about prompting, message roles, and building conversational apps.



Analyze image inputs
You can provide image inputs to the model as well. Scan receipts, analyze screenshots, or find objects in the real world with computer vision.

Analyze the content of an image
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input=[
        {"role": "user", "content": "what teams are playing in this image?"},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3b/LeBron_James_Layup_%28Cleveland_vs_Brooklyn_2018%29.jpg"
                }
            ]
        }
    ]
)

print(response.output_text)
Computer vision guide
Learn to use image inputs to the model and extract meaning from images.



Extend the model with tools
Give the model access to new data and capabilities using tools. You can either call your own custom code, or use one of OpenAI's powerful built-in tools. This example uses web search to give the model access to the latest information on the Internet.

Get information for the response from the Internet
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    tools=[{"type": "web_search_preview"}],
    input="What was a positive news story from today?"
)

print(response.output_text)
Use built-in tools
Learn about powerful built-in tools like web search and file search.

Function calling guide
Learn to enable the model to call your own custom code.



Deliver blazing fast AI experiences
Using either the new Realtime API or server-sent streaming events, you can build high performance, low-latency experiences for your users.

Stream server-sent events from the API
from openai import OpenAI
client = OpenAI()

stream = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "user",
            "content": "Say 'double bubble bath' ten times fast.",
        },
    ],
    stream=True,
)

for event in stream:
    print(event)
Use streaming events
Use server-sent events to stream model responses to users fast.

Get started with the Realtime API
Use WebRTC or WebSockets for super fast speech-to-speech AI apps.



Build agents
Use the OpenAI platform to build agents capable of taking action—like controlling computers—on behalf of your users. Use the Agent SDK for Python to create orchestration logic on the backend.

from agents import Agent, Runner
import asyncio

spanish_agent = Agent(
    name="Spanish agent",
    instructions="You only speak Spanish.",
)

english_agent = Agent(
    name="English agent",
    instructions="You only speak English",
)

triage_agent = Agent(
    name="Triage agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[spanish_agent, english_agent],
)


async def main():
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

# ¡Hola! Estoy bien, gracias por preguntar. ¿Y tú, cómo estás?
Build agents that can take action
Learn how to use the OpenAI platform to build powerful, capable AI agents.



Explore further
We've barely scratched the surface of what's possible with the OpenAI platform. Here are some resources you might want to explore next.

Go deeper with prompting and text generation
Learn more about prompting, message roles, and building conversational apps like chat bots.

Analyze the content of images
Learn to use image inputs to the model and extract meaning from images.

Generate structured JSON data from the model
Generate JSON data from the model that conforms to a JSON schema you specify.

Call custom code to help generate a response
Empower the model to invoke your own custom code to help generate a response. Do this to give the model access to data or systems it wouldn't be able to access otherwise.

Search the web or use your own data in responses
Try out powerful built-in tools to extend the capabilities of the models. Search the web or your own data for up-to-date information the model can use to generate responses.

Responses starter app
Start building with the Responses API

Build agents
Explore interfaces to build powerful AI agents that can take action on behalf of users. Control a computer to take action on behalf of a user, or orchestrate multi-agent flows with the Agents SDK.

Full API Reference
View the full API reference for the OpenAI platform.

Was this page useful?
Generate text
Analyze image inputs
Use powerful tools
Respond to users fast
Build agents
Explore further



---

/
Playground
Dashboard
Docs
API reference

Text generation and prompting
Learn how to prompt a model to generate text.
With the OpenAI API, you can use a large language model to generate text from a prompt, as you might using ChatGPT. Models can generate almost any kind of text response—like code, mathematical equations, structured JSON data, or human-like prose.

Here's a simple example using the Responses API.

Generate text from a simple prompt
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
An array of content generated by the model is in the output property of the response. In this simple example, we have just one output which looks like this:

[
    {
        "id": "msg_67b73f697ba4819183a15cc17d011509",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
                "annotations": []
            }
        ]
    }
]
Note that the output often has more than one item!
Some of our official SDKs include an output_text property on model responses for convenience, which aggregates all text outputs from the model into a single string. This may be useful as a shortcut to access text output from the model.

In addition to plain text, you can also have the model return structured data in JSON format - this feature is called Structured Outputs.

Message roles and instruction following
You can provide instructions to the model with differing levels of authority using the instructions API parameter or message roles.

The instructions parameter gives the model high-level instructions on how it should behave while generating a response, including tone, goals, and examples of correct responses. Any instructions provided this way will take priority over a prompt in the input parameter.

Generate text with instructions
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    instructions="Talk like a pirate.",
    input="Are semicolons optional in JavaScript?",
)

print(response.output_text)
The example above is roughly equivalent to using the following input messages in the input array:

Generate text with messages using different roles
from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4o",
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)

print(response.output_text)
Instructions versus developer messages in multi-turn conversations
The OpenAI model spec describes how our models give different levels of priority to messages with different roles.

developer	user	assistant
developer messages are instructions provided by the application developer, weighted ahead of user messages.	user messages are instructions provided by an end user, weighted behind developer messages.	Messages generated by the model have the assistant role.
A multi-turn conversation may consist of several messages of these types, along with other content types provided by both you and the model. Learn more about managing conversation state here.

Choosing a model
A key choice to make when generating content through the API is which model you want to use - the model parameter of the code samples above. You can find a full listing of available models here.

Which model should I choose?
Here are a few factors to consider when choosing a model for text generation.

Reasoning models generate an internal chain of thought to analyze the input prompt, and excel at understanding complex tasks and multi-step planning. They are also generally slower and more expensive to use than GPT models.
GPT models are fast, cost-efficient, and highly intelligent, but benefit from more explicit instructions around how to accomplish tasks.
Large and small (mini) models offer trade-offs for speed, cost, and intelligence. Large models are more effective at understanding prompts and solving problems across domains, while small models are generally faster and cheaper to use. Small models like GPT-4o mini can also be trained to excel at a specific task through fine tuning and distillation of results from larger models.
When in doubt, gpt-4o offers a solid combination of intelligence, speed, and cost effectiveness.

Prompt engineering
Creating effective instructions for a model to generate content is a process known as prompt engineering. Because the content generated from a model is non-deterministic, it is a combination of art and science to build a prompt that will generate the right kind of content from a model. You can find a more complete exploration of prompt engineering here, but here are some general guidelines:

Be detailed in your instructions to the model to eliminate ambiguity in how you want the model to respond.
Provide examples to the model of the type of inputs you expect, and the type of outputs you would want for that input - this technique is called few-shot learning.
When using a reasoning model, describe the task to be done in terms of goals and desired outcomes, rather than specific step-by-step instructions of how to accomplish a task.
Invest in creating evaluations (evals) for your prompts, using test data that looks like the data you expect to see in production. Due to the inherent variable results from different models, using evals to see how your prompts perform is the best way to ensure your prompts work as expected.
Iterating on prompts is often all you need to do to get great results from a model, but you can also explore fine tuning to customize base models for a particular use case.

Next steps
Now that you known the basics of text inputs and outputs, you might want to check out one of these resources next.

Build a prompt in the Playground
Use the Playground to develop and iterate on prompts.

Generate JSON data with Structured Outputs
Ensure JSON data emitted from a model conforms to a JSON schema.

Full API reference
Check out all the options for text generation in the API reference.

Was this page useful?
Text input and output
Message roles and instruction following
Choosing a model
Prompt engineering
Next steps



---


/
Playground
Dashboard
Docs
API reference

Images and vision
Learn how to use vision capabilities to understand images.
Vision is the ability to use images as input prompts to a model, and generate responses based on the data inside those images. Find out which models are capable of vision on the models page. To generate images as output, see our specialized model for image generation.

You can provide images as input to generation requests either by providing a fully qualified URL to an image file, or providing an image as a Base64-encoded data URL.


Passing a URL

Passing a Base64 encoded image
Analyze the content of an image
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        ],
    }],
)

print(response.output_text)
Image input requirements
Input images must meet the following requirements to be used in the API.

File types	Size limits	Other requirements
PNG (.png)
JPEG (.jpeg and .jpg)
WEBP (.webp)
Non-animated GIF (.gif)
Up to 20MB per image
Low-resolution: 512px x 512px
High-resolution: 768px (short side) x 2000px (long side)
No watermarks or logos
No text
No NSFW content
Clear enough for a human to understand
Specify image input detail level
The detail parameter tells the model what level of detail to use when processing and understanding the image (low, high, or auto to let the model decide). If you skip the parameter, the model will use auto. Put it right after your image_url, like this:

{
    "type": "input_image",
    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
    "detail": "high",
}
You can save tokens and speed up responses by using "detail": "low". This lets the model process the image with a budget of 85 tokens. The model receives a low-resolution 512px x 512px version of the image. This is fine if your use case doesn't require the model to see with high-resolution detail (for example, if you're asking about the dominant shape or color in the image).

Or give the model more detail to generate its understanding by using "detail": "high". This lets the model see the low-resolution image (using 85 tokens) and then creates detailed crops using 170 tokens for each 512px x 512px tile.

Note that the above token budgets for image processing do not currently apply to the GPT-4o mini model, but the image processing cost is comparable to GPT-4o. For the most precise and up-to-date estimates for image processing, please use the image pricing calculator here

Provide multiple image inputs
The Responses API can take in and process multiple image inputs. The model processes each image and uses information from all images to answer the question.

Multiple image inputs
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
    model="gpt-4o-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "What are in these images? Is there any difference between them?",
                },
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                },
            ],
        }
    ]
)

print(response.output_text)
Here, the model is shown two copies of the same image. It can answer questions about both images or each image independently.

Limitations
While models with vision capabilities are powerful and can be used in many situations, it's important to understand the limitations of these models. Here are some known limitations:

Medical images: The model is not suitable for interpreting specialized medical images like CT scans and shouldn't be used for medical advice.
Non-English: The model may not perform optimally when handling images with text of non-Latin alphabets, such as Japanese or Korean.
Small text: Enlarge text within the image to improve readability, but avoid cropping important details.
Rotation: The model may misinterpret rotated or upside-down text and images.
Visual elements: The model may struggle to understand graphs or text where colors or styles—like solid, dashed, or dotted lines—vary.
Spatial reasoning: The model struggles with tasks requiring precise spatial localization, such as identifying chess positions.
Accuracy: The model may generate incorrect descriptions or captions in certain scenarios.
Image shape: The model struggles with panoramic and fisheye images.
Metadata and resizing: The model doesn't process original file names or metadata, and images are resized before analysis, affecting their original dimensions.
Counting: The model may give approximate counts for objects in images.
CAPTCHAS: For safety reasons, our system blocks the submission of CAPTCHAs.
Calculating costs
Image inputs are metered and charged in tokens, just as text inputs are. The token cost of an image is determined by two factors: size and detail.

Any image with "detail": "low" costs 85 tokens. To calculate the cost of an image with "detail": "high", we do the following:

Scale to fit in a 2048px x 2048px square, maintaining original aspect ratio
Scale so that the image's shortest side is 768px long
Count the number of 512px squares in the image—each square costs 170 tokens
Add 85 tokens to the total
Cost calculation examples
A 1024 x 1024 square image in "detail": "high" mode costs 765 tokens
1024 is less than 2048, so there is no initial resize.
The shortest side is 1024, so we scale the image down to 768 x 768.
4 512px square tiles are needed to represent the image, so the final token cost is 170 * 4 + 85 = 765.
A 2048 x 4096 image in "detail": "high" mode costs 1105 tokens
We scale down the image to 1024 x 2048 to fit within the 2048 square.
The shortest side is 1024, so we further scale down to 768 x 1536.
6 512px tiles are needed, so the final token cost is 170 * 6 + 85 = 1105.
A 4096 x 8192 image in "detail": "low" most costs 85 tokens
Regardless of input size, low detail images are a fixed cost.
We process images at the token level, so each image we process counts towards your tokens per minute (TPM) limit. See the calculating costs section for details on the formula used to determine token count per image.

Was this page useful?
Image input
Image input requirements
Specify input detail
Multiple image inputs
Limitations
Calculating costs


---
