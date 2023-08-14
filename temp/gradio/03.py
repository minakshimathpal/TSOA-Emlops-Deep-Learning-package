from gradio_client import Client

client = Client("https://7860-minakshimat-tsoaemlopsd-swngm0koccp.ws-us103.gitpod.io/")
result = client.predict(
				"https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",	# str (filepath or URL to image) 								in 'inp_img' Image component
				api_name="/predict"
)
print(result)