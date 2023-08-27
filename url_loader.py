from fastapi import FastAPI
from playwright.async_api import async_playwright

app = FastAPI()

async def get_data_from_url():
    urls = [
        "https://www.youtube.com/watch?v=dzXZg7Ki7AY",
    ]
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(urls[0])
        content = await page.content()
        await browser.close()
        return content

@app.get("/")
async def read_root():
    data = await get_data_from_url()
    return {"data": data}
