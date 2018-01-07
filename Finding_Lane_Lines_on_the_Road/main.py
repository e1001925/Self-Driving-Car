from frame_pipeline import *

if __name__ == '__main__':
    decision = input("Enter test target 1.images, 2.video : ")
    is_solid = input("Do you want it is solid? yes/no : ")
    if(is_solid == 'yes'):
        is_solid = 1
    else:
        is_solid = 0
    if decision[0] == '1':
        print("Images will open soon...")
        on_image(is_solid)
    elif decision[0] == '2':
        print("Video will open soon...")
        on_video(is_solid)
    else:
        print("Unexpected answer! You are kidding me, good bye")
        
    

