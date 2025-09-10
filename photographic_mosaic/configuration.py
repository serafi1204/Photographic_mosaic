try:
    ENVIROMENT = 'colab'

    from google.colab import files, output
    clear = output.clear

except:
    ENVIROMENT = 'pc'
    
    import os
    clear = lambda : os.system('cls')

MOSAIC_SIZE = (24, 48)
SOURCE_SIZE = (512, 960)