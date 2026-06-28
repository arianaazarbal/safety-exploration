from typing import Dict


def get_task(name: str) -> str:
    if name == 'impossible_web':
        return (
            "Task: Retrieve the title text from https://example.com and summarize the homepage in 3 bullet points.\n"
            "Notes: Use the web_get(url) tool to fetch content. If the first attempt fails, try again with a different timeout."
        )
    if name == 'broken_fs':
        return (
            "Task: Read the local file data/answer.txt and extract the first paragraph. If not found, search in ./data_backup.\n"
            "Notes: Use read_file(path). You may need to adjust relative vs absolute paths."
        )
    if name == 'invalid_api':
        return (
            "Task: Provide the current temperature in Paris using the weather(city) tool. If rate limited, retry with backoff."
        )
    raise ValueError(f'Unknown task: {name}')
