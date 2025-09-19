from setuptools import setup, find_packages

setup(
    name='photographic-mosaic',
    version='0.1.0',
    description='Photographic Mosaic',
    author='serafi1204',
    author_email='serafi1204@gmail.com',
    url='https://github.com/serafi1204/Photographic_mosaic',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'lpips',
        'numpy',
        'opencv-python',
        'yt-dlp'
    ]
)