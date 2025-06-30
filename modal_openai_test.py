# Minimal Modal test to check openai version
import modal

app = modal.App("openai-version-test")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "openai==1.2.4"
)

@app.function(image=image)
def print_openai_version():
    import openai
    print("OpenAI version:", openai.__version__)

if __name__ == "__main__":
    print_openai_version.remote() 