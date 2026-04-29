#link a la imagen: https://drive.google.com/file/d/1hMgboSGTSECWoqNfGDEviki1MSF8BjwB/view?usp=sharing
import py5

def setup():
    py5.size(600,600)
    

def draw():
    img = py5.load_image("C:/Users/cin_c/OneDrive/Desktop/Repositorios/pdi/coronel-cintia-pdi-1c-2026/001/003 - LAB/imagen 01.png")
    py5.background(255)
    py5.image(img, 100, 100, 400, 400)  
    py5.apply_filter(py5.BLUR, 1)
    py5.apply_filter(py5.DILATE)
    py5.apply_filter(py5.INVERT)
    
    if py5.is_mouse_pressed:
        print("Esta es una solarigrafía!")
        py5.fill(255, 0, 0)
        py5.circle(py5.mouse_x, py5.mouse_y, 40)

    #podría haber hecho que al hacer clic por fuera de la imagen salte un mensaje, pero me trabé con el inverter (lo tenía por debajo del if antes)
    #así que no llegué a corregirlo

py5.run_sketch()
