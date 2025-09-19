try:
    ENVIROMENT = 'colab'

    from google.colab import files, output
    clear_cmd = output.clear

except:
    ENVIROMENT = 'pc'
    
    import os
    clear_cmd = lambda : os.system('cls')

MOSAIC_SIZE = (16, 30)
SOURCE_SIZE = (512, 960) 