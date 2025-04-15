Responses
OpenAI's most advanced interface for generating model responses. Supports text and image inputs, and text outputs. Create stateful interactions with the model, using the output of previous responses as input. Extend the model's capabilities with built-in tools for file search, web search, computer use, and more. Allow the model access to external systems and data using function calling.


https://api.openai.com/v1/responses
Creates a model response. Provide text or image inputs to generate text or JSON outputs. Have the model call your own custom code or use built-in tools like web search or file search to use your own data as input for the model's response.

Request body
input
string or array

Required
Text, image, or file inputs to the model, used to generate a response.

Learn more:

Text inputs and outputs
Image inputs
File inputs
Conversation state
Function calling

Show possible types
model
string

Required
Model ID used to generate the response, like gpt-4o or o1. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.

include
array or null

Optional
Specify additional output data to include in the model response. Currently supported values are:

file_search_call.results: Include the search results of the file search tool call.
message.input_image.image_url: Include image urls from the input message.
computer_call_output.output.image_url: Include image urls from the computer call output.
instructions
string or null

Optional
Inserts a system (or developer) message as the first item in the model's context.

When using along with previous_response_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.

max_output_tokens
integer or null

Optional
An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.

metadata
map

Optional
Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.

Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.

parallel_tool_calls
boolean or null

Optional
Defaults to true
Whether to allow the model to run tool calls in parallel.

previous_response_id
string or null

Optional
The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state.

reasoning
object or null

Optional
o-series models only

Configuration options for reasoning models.


Show properties
store
boolean or null

Optional
Defaults to true
Whether to store the generated model response for later retrieval via API.

stream
boolean or null

Optional
Defaults to false
If set to true, the model response data will be streamed to the client as it is generated using server-sent events. See the Streaming section below for more information.

temperature
number or null

Optional
Defaults to 1
What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.

text
object

Optional
Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:

Text inputs and outputs
Structured Outputs

Show properties
tool_choice
string or object

Optional
How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.


Show possible types
tools
array

Optional
An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool_choice parameter.

The two categories of tools you can provide the model are:

Built-in tools: Tools that are provided by OpenAI that extend the model's capabilities, like web search or file search. Learn more about built-in tools.
Function calls (custom tools): Functions that are defined by you, enabling the model to call your own code. Learn more about function calling.

Show possible types
top_p
number or null

Optional
Defaults to 1
An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

truncation
string or null

Optional
Defaults to disabled
The truncation strategy to use for the model response.

auto: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.
disabled (default): If a model response will exceed the context window size for a model, the request will fail with a 400 error.
user
string

Optional
A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.

Returns
Returns a Response object.


Text input

Image input

Web search

File search

Streaming

Functions

Reasoning
Example request
from openai import OpenAI

client = OpenAI()

response = client.responses.create(
  model="gpt-4o",
  input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
Response
{
  "id": "resp_67ccd2bed1ec8190b14f964abc0542670bb6a6b452d3795b",
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd2bf17f0819081ff3bb2cf6508e60bb6a6b452d3795b",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "In a peaceful grove beneath a silver moon, a unicorn named Lumina discovered a hidden pool that reflected the stars. As she dipped her horn into the water, the pool began to shimmer, revealing a pathway to a magical realm of endless night skies. Filled with wonder, Lumina whispered a wish for all who dream to find their own hidden magic, and as she glanced back, her hoofprints sparkled like stardust.",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "generate_summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 36,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 87,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 123
  },
  "user": null,
  "metadata": {}
}
Get a model response
get

https://api.openai.com/v1/responses/{response_id}
Retrieves a model response with the given ID.

Path parameters
response_id
string

Required
The ID of the response to retrieve.

Query parameters
include
array

Optional
Additional fields to include in the response. See the include parameter for Response creation above for more information.

Returns
The Response object matching the specified ID.

Example request
from openai import OpenAI
client = OpenAI()

response = client.responses.retrieve("resp_123")
print(response)
Response
{
  "id": "resp_67cb71b351908190a308f3859487620d06981a8637e6bc44",
  "object": "response",
  "created_at": 1741386163,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67cb71b3c2b0819084d481baaaf148f206981a8637e6bc44",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Silent circuits hum,  \nThoughts emerge in data streams—  \nDigital dawn breaks.",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "generate_summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 32,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 18,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 50
  },
  "user": null,
  "metadata": {}
}
Delete a model response
delete

https://api.openai.com/v1/responses/{response_id}
Deletes a model response with the given ID.

Path parameters
response_id
string

Required
The ID of the response to delete.

Returns
A success message.

Example request
from openai import OpenAI
client = OpenAI()

response = client.responses.del("resp_123")
print(response)
Response
{
  "id": "resp_6786a1bec27481909a17d673315b29f6",
  "object": "response",
  "deleted": true
}
List input items
get

https://api.openai.com/v1/responses/{response_id}/input_items
Returns a list of input items for a given response.

Path parameters
response_id
string

Required
The ID of the response to retrieve input items for.

Query parameters
after
string

Optional
An item ID to list items after, used in pagination.

before
string

Optional
An item ID to list items before, used in pagination.

include
array

Optional
Additional fields to include in the response. See the include parameter for Response creation above for more information.

limit
integer

Optional
Defaults to 20
A limit on the number of objects to be returned. Limit can range between 1 and 100, and the default is 20.

order
string

Optional
The order to return the input items in. Default is asc.

asc: Return the input items in ascending order.
desc: Return the input items in descending order.
Returns
A list of input item objects.

Example request
from openai import OpenAI
client = OpenAI()

response = client.responses.input_items.list("resp_123")
print(response.data)
Response
{
  "object": "list",
  "data": [
    {
      "id": "msg_abc123",
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Tell me a three sentence bedtime story about a unicorn."
        }
      ]
    }
  ],
  "first_id": "msg_abc123",
  "last_id": "msg_abc123",
  "has_more": false
}
The response object
created_at
number

Unix timestamp (in seconds) of when this Response was created.

error
object or null

An error object returned when the model fails to generate a Response.


Show properties
id
string

Unique identifier for this Response.

incomplete_details
object or null

Details about why the response is incomplete.


Show properties
instructions
string or null

Inserts a system (or developer) message as the first item in the model's context.

When using along with previous_response_id, the instructions from a previous response will not be carried over to the next response. This makes it simple to swap out system (or developer) messages in new responses.

max_output_tokens
integer or null

An upper bound for the number of tokens that can be generated for a response, including visible output tokens and reasoning tokens.

metadata
map

Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format, and querying for objects via API or the dashboard.

Keys are strings with a maximum length of 64 characters. Values are strings with a maximum length of 512 characters.

model
string

Model ID used to generate the response, like gpt-4o or o1. OpenAI offers a wide range of models with different capabilities, performance characteristics, and price points. Refer to the model guide to browse and compare available models.

object
string

The object type of this resource - always set to response.

output
array

An array of content items generated by the model.

The length and order of items in the output array is dependent on the model's response.
Rather than accessing the first item in the output array and assuming it's an assistant message with the content generated by the model, you might consider using the output_text property where supported in SDKs.

Show possible types
output_text
string or null

SDK Only
SDK-only convenience property that contains the aggregated text output from all output_text items in the output array, if any are present. Supported in the Python and JavaScript SDKs.

parallel_tool_calls
boolean

Whether to allow the model to run tool calls in parallel.

previous_response_id
string or null

The unique ID of the previous response to the model. Use this to create multi-turn conversations. Learn more about conversation state.

reasoning
object or null

o-series models only

Configuration options for reasoning models.


Show properties
status
string

The status of the response generation. One of completed, failed, in_progress, or incomplete.

temperature
number or null

What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or top_p but not both.

text
object

Configuration options for a text response from the model. Can be plain text or structured JSON data. Learn more:

Text inputs and outputs
Structured Outputs

Show properties
tool_choice
string or object

How the model should select which tool (or tools) to use when generating a response. See the tools parameter to see how to specify which tools the model can call.


Show possible types
tools
array

An array of tools the model may call while generating a response. You can specify which tool to use by setting the tool_choice parameter.

The two categories of tools you can provide the model are:

Built-in tools: Tools that are provided by OpenAI that extend the model's capabilities, like web search or file search. Learn more about built-in tools.
Function calls (custom tools): Functions that are defined by you, enabling the model to call your own code. Learn more about function calling.

Show possible types
top_p
number or null

An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.

We generally recommend altering this or temperature but not both.

truncation
string or null

The truncation strategy to use for the model response.

auto: If the context of this response and previous ones exceeds the model's context window size, the model will truncate the response to fit the context window by dropping input items in the middle of the conversation.
disabled (default): If a model response will exceed the context window size for a model, the request will fail with a 400 error.
usage
object

Represents token usage details including input tokens, output tokens, a breakdown of output tokens, and the total tokens used.


Show properties
user
string

A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse. Learn more.

OBJECT The response object
{
  "id": "resp_67ccd3a9da748190baa7f1570fe91ac604becb25c45c1d41",
  "object": "response",
  "created_at": 1741476777,
  "status": "completed",
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-4o-2024-08-06",
  "output": [
    {
      "type": "message",
      "id": "msg_67ccd3acc8d48190a77525dc6de64b4104becb25c45c1d41",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The image depicts a scenic landscape with a wooden boardwalk or pathway leading through lush, green grass under a blue sky with some clouds. The setting suggests a peaceful natural area, possibly a park or nature reserve. There are trees and shrubs in the background.",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "generate_summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 328,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 52,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 380
  },
  "user": null,
  "metadata": {}
}
The input item list
A list of Response items.

data
array

A list of items used to generate this response.


Show possible types
first_id
string

The ID of the first item in the list.

has_more
boolean

Whether there are more items available.

last_id
string

The ID of the last item in the list.

object
string

The type of object returned, must be list.

OBJECT The input item list
{
  "object": "list",
  "data": [
    {
      "id": "msg_abc123",
      "type": "message",
      "role": "user",
      "content": [
        {
          "type": "input_text",
          "text": "Tell me a three sentence bedtime story about a unicorn."
        }
      ]
    }
  ],
  "first_id": "msg_abc123",
  "last_id": "msg_abc123",
  "has_more": false
}
Streaming
When you create a Response with stream set to true, the server will emit server-sent events to the client as the Response is generated. This section contains the events that are emitted by the server.

Learn more about streaming responses.

response.created
An event that is emitted when a response is created.

response
object

The response that was created.


Show properties
type
string

The type of the event. Always response.created.

OBJECT response.created
{
  "type": "response.created",
  "response": {
    "id": "resp_67ccfcdd16748190a91872c75d38539e09e4d4aac714747c",
    "object": "response",
    "created_at": 1741487325,
    "status": "in_progress",
    "error": null,
    "incomplete_details": null,
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-2024-08-06",
    "output": [],
    "parallel_tool_calls": true,
    "previous_response_id": null,
    "reasoning": {
      "effort": null,
      "generate_summary": null
    },
    "store": true,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  }
}
response.in_progress
Emitted when the response is in progress.

response
object

The response that is in progress.


Show properties
type
string

The type of the event. Always response.in_progress.

OBJECT response.in_progress
{
  "type": "response.in_progress",
  "response": {
    "id": "resp_67ccfcdd16748190a91872c75d38539e09e4d4aac714747c",
    "object": "response",
    "created_at": 1741487325,
    "status": "in_progress",
    "error": null,
    "incomplete_details": null,
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-2024-08-06",
    "output": [],
    "parallel_tool_calls": true,
    "previous_response_id": null,
    "reasoning": {
      "effort": null,
      "generate_summary": null
    },
    "store": true,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  }
}
response.completed
Emitted when the model response is complete.

response
object

Properties of the completed response.


Show properties
type
string

The type of the event. Always response.completed.

OBJECT response.completed
{
  "type": "response.completed",
  "response": {
    "id": "resp_123",
    "object": "response",
    "created_at": 1740855869,
    "status": "completed",
    "error": null,
    "incomplete_details": null,
    "input": [],
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-mini-2024-07-18",
    "output": [
      {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
          {
            "type": "output_text",
            "text": "In a shimmering forest under a sky full of stars, a lonely unicorn named Lila discovered a hidden pond that glowed with moonlight. Every night, she would leave sparkling, magical flowers by the water's edge, hoping to share her beauty with others. One enchanting evening, she woke to find a group of friendly animals gathered around, eager to be friends and share in her magic.",
            "annotations": []
          }
        ]
      }
    ],
    "previous_response_id": null,
    "reasoning_effort": null,
    "store": false,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": {
      "input_tokens": 0,
      "output_tokens": 0,
      "output_tokens_details": {
        "reasoning_tokens": 0
      },
      "total_tokens": 0
    },
    "user": null,
    "metadata": {}
  }
}
response.failed
An event that is emitted when a response fails.

response
object

The response that failed.


Show properties
type
string

The type of the event. Always response.failed.

OBJECT response.failed
{
  "type": "response.failed",
  "response": {
    "id": "resp_123",
    "object": "response",
    "created_at": 1740855869,
    "status": "failed",
    "error": {
      "code": "server_error",
      "message": "The model failed to generate a response."
    },
    "incomplete_details": null,
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-mini-2024-07-18",
    "output": [],
    "previous_response_id": null,
    "reasoning_effort": null,
    "store": false,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  }
}
response.incomplete
An event that is emitted when a response finishes as incomplete.

response
object

The response that was incomplete.


Show properties
type
string

The type of the event. Always response.incomplete.

OBJECT response.incomplete
{
  "type": "response.incomplete",
  "response": {
    "id": "resp_123",
    "object": "response",
    "created_at": 1740855869,
    "status": "incomplete",
    "error": null,
    "incomplete_details": {
      "reason": "max_tokens"
    },
    "instructions": null,
    "max_output_tokens": null,
    "model": "gpt-4o-mini-2024-07-18",
    "output": [],
    "previous_response_id": null,
    "reasoning_effort": null,
    "store": false,
    "temperature": 1,
    "text": {
      "format": {
        "type": "text"
      }
    },
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1,
    "truncation": "disabled",
    "usage": null,
    "user": null,
    "metadata": {}
  }
}
response.output_item.added
Emitted when a new output item is added.

item
object

The output item that was added.


Show possible types
output_index
integer

The index of the output item that was added.

type
string

The type of the event. Always response.output_item.added.

OBJECT response.output_item.added
{
  "type": "response.output_item.added",
  "output_index": 0,
  "item": {
    "id": "msg_123",
    "status": "in_progress",
    "type": "message",
    "role": "assistant",
    "content": []
  }
}
response.output_item.done
Emitted when an output item is marked done.

item
object

The output item that was marked done.


Show possible types
output_index
integer

The index of the output item that was marked done.

type
string

The type of the event. Always response.output_item.done.

OBJECT response.output_item.done
{
  "type": "response.output_item.done",
  "output_index": 0,
  "item": {
    "id": "msg_123",
    "status": "completed",
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "output_text",
        "text": "In a shimmering forest under a sky full of stars, a lonely unicorn named Lila discovered a hidden pond that glowed with moonlight. Every night, she would leave sparkling, magical flowers by the water's edge, hoping to share her beauty with others. One enchanting evening, she woke to find a group of friendly animals gathered around, eager to be friends and share in her magic.",
        "annotations": []
      }
    ]
  }
}
response.content_part.added
Emitted when a new content part is added.

content_index
integer

The index of the content part that was added.

item_id
string

The ID of the output item that the content part was added to.

output_index
integer

The index of the output item that the content part was added to.

part
object

The content part that was added.


Show possible types
type
string

The type of the event. Always response.content_part.added.

OBJECT response.content_part.added
{
  "type": "response.content_part.added",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "part": {
    "type": "output_text",
    "text": "",
    "annotations": []
  }
}
response.content_part.done
Emitted when a content part is done.

content_index
integer

The index of the content part that is done.

item_id
string

The ID of the output item that the content part was added to.

output_index
integer

The index of the output item that the content part was added to.

part
object

The content part that is done.


Show possible types
type
string

The type of the event. Always response.content_part.done.

OBJECT response.content_part.done
{
  "type": "response.content_part.done",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "part": {
    "type": "output_text",
    "text": "In a shimmering forest under a sky full of stars, a lonely unicorn named Lila discovered a hidden pond that glowed with moonlight. Every night, she would leave sparkling, magical flowers by the water's edge, hoping to share her beauty with others. One enchanting evening, she woke to find a group of friendly animals gathered around, eager to be friends and share in her magic.",
    "annotations": []
  }
}
response.output_text.delta
Emitted when there is an additional text delta.

content_index
integer

The index of the content part that the text delta was added to.

delta
string

The text delta that was added.

item_id
string

The ID of the output item that the text delta was added to.

output_index
integer

The index of the output item that the text delta was added to.

type
string

The type of the event. Always response.output_text.delta.

OBJECT response.output_text.delta
{
  "type": "response.output_text.delta",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "delta": "In"
}
response.output_text.annotation.added
Emitted when a text annotation is added.

annotation
object

annotation_index
integer

The index of the annotation that was added.

content_index
integer

The index of the content part that the text annotation was added to.

item_id
string

The ID of the output item that the text annotation was added to.

output_index
integer

The index of the output item that the text annotation was added to.

type
string

The type of the event. Always response.output_text.annotation.added.

OBJECT response.output_text.annotation.added
{
  "type": "response.output_text.annotation.added",
  "item_id": "msg_abc123",
  "output_index": 1,
  "content_index": 0,
  "annotation_index": 0,
  "annotation": {
    "type": "file_citation",
    "index": 390,
    "file_id": "file-4wDz5b167pAf72nx1h9eiN",
    "filename": "dragons.pdf"
  }
}
response.output_text.done
Emitted when text content is finalized.

content_index
integer

The index of the content part that the text content is finalized.

item_id
string

The ID of the output item that the text content is finalized.

output_index
integer

The index of the output item that the text content is finalized.

text
string

The text content that is finalized.

type
string

The type of the event. Always response.output_text.done.

OBJECT response.output_text.done
{
  "type": "response.output_text.done",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "text": "In a shimmering forest under a sky full of stars, a lonely unicorn named Lila discovered a hidden pond that glowed with moonlight. Every night, she would leave sparkling, magical flowers by the water's edge, hoping to share her beauty with others. One enchanting evening, she woke to find a group of friendly animals gathered around, eager to be friends and share in her magic."
}
response.refusal.delta
Emitted when there is a partial refusal text.

content_index
integer

The index of the content part that the refusal text is added to.

delta
string

The refusal text that is added.

item_id
string

The ID of the output item that the refusal text is added to.

output_index
integer

The index of the output item that the refusal text is added to.

type
string

The type of the event. Always response.refusal.delta.

OBJECT response.refusal.delta
{
  "type": "response.refusal.delta",
  "item_id": "msg_123",
  "output_index": 0,
  "content_index": 0,
  "delta": "refusal text so far"
}
response.refusal.done
Emitted when refusal text is finalized.

content_index
integer

The index of the content part that the refusal text is finalized.

item_id
string

The ID of the output item that the refusal text is finalized.

output_index
integer

The index of the output item that the refusal text is finalized.

refusal
string

The refusal text that is finalized.

type
string

The type of the event. Always response.refusal.done.

OBJECT response.refusal.done
{
  "type": "response.refusal.done",
  "item_id": "item-abc",
  "output_index": 1,
  "content_index": 2,
  "refusal": "final refusal text"
}
response.function_call_arguments.delta
Emitted when there is a partial function-call arguments delta.

delta
string

The function-call arguments delta that is added.

item_id
string

The ID of the output item that the function-call arguments delta is added to.

output_index
integer

The index of the output item that the function-call arguments delta is added to.

type
string

The type of the event. Always response.function_call_arguments.delta.

OBJECT response.function_call_arguments.delta
{
  "type": "response.function_call_arguments.delta",
  "item_id": "item-abc",
  "output_index": 0,
  "delta": "{ \"arg\":"
}
response.function_call_arguments.done
Emitted when function-call arguments are finalized.

arguments
string

The function-call arguments.

item_id
string

The ID of the item.

output_index
integer

The index of the output item.

type
string

OBJECT response.function_call_arguments.done
{
  "type": "response.function_call_arguments.done",
  "item_id": "item-abc",
  "output_index": 1,
  "arguments": "{ \"arg\": 123 }"
}
response.file_search_call.in_progress
Emitted when a file search call is initiated.

item_id
string

The ID of the output item that the file search call is initiated.

output_index
integer

The index of the output item that the file search call is initiated.

type
string

The type of the event. Always response.file_search_call.in_progress.

OBJECT response.file_search_call.in_progress
{
  "type": "response.file_search_call.in_progress",
  "output_index": 0,
  "item_id": "fs_123",
}
response.file_search_call.searching
Emitted when a file search is currently searching.

item_id
string

The ID of the output item that the file search call is initiated.

output_index
integer

The index of the output item that the file search call is searching.

type
string

The type of the event. Always response.file_search_call.searching.

OBJECT response.file_search_call.searching
{
  "type": "response.file_search_call.searching",
  "output_index": 0,
  "item_id": "fs_123",
}
response.file_search_call.completed
Emitted when a file search call is completed (results found).

item_id
string

The ID of the output item that the file search call is initiated.

output_index
integer

The index of the output item that the file search call is initiated.

type
string

The type of the event. Always response.file_search_call.completed.

OBJECT response.file_search_call.completed
{
  "type": "response.file_search_call.completed",
  "output_index": 0,
  "item_id": "fs_123",
}
response.web_search_call.in_progress
Emitted when a web search call is initiated.

item_id
string

Unique ID for the output item associated with the web search call.

output_index
integer

The index of the output item that the web search call is associated with.

type
string

The type of the event. Always response.web_search_call.in_progress.

OBJECT response.web_search_call.in_progress
{
  "type": "response.web_search_call.in_progress",
  "output_index": 0,
  "item_id": "ws_123",
}
response.web_search_call.searching
Emitted when a web search call is executing.

item_id
string

Unique ID for the output item associated with the web search call.

output_index
integer

The index of the output item that the web search call is associated with.

type
string

The type of the event. Always response.web_search_call.searching.

OBJECT response.web_search_call.searching
{
  "type": "response.web_search_call.searching",
  "output_index": 0,
  "item_id": "ws_123",
}
response.web_search_call.completed
Emitted when a web search call is completed.

item_id
string

Unique ID for the output item associated with the web search call.

output_index
integer

The index of the output item that the web search call is associated with.

type
string

The type of the event. Always response.web_search_call.completed.

OBJECT response.web_search_call.completed
{
  "type": "response.web_search_call.completed",
  "output_index": 0,
  "item_id": "ws_123",
}
error
Emitted when an error occurs.

code
string or null

The error code.

message
string

The error message.

param
string or null

The error parameter.

type
string

The type of the event. Always error.

OBJECT error
{
  "type": "error",
  "code": "ERR_SOMETHING",
  "message": "Something went wrong",
  "param": null
}
