import wget
import os

output_directory = "./datasets/open_images/test_A"
if not os.path.exists(output_directory):
	os.makedirs(output_directory)


for i in range(25):
	url = "http://www.cs.albany.edu/~xypan/research/img/Kodak/kodim{:02d}.png".format(i+1)
	f = wget.download(url, out=output_directory)
	