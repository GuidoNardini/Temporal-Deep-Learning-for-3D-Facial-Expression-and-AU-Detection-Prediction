import cv2
import os

main_folder = '/andromeda/personal/fgastaldi/facemorphic_output/53189cf9202642bd917f31d6eabf8053/AU_51/frames_1' 
# Cartella contenente i frames da montare

frame_files = [] # Lista di tutti i percorsi dei frames
for subfolder in os.listdir(main_folder):
    subfolder_path = os.path.join(main_folder, subfolder)
    if os.path.isdir(subfolder_path):
        image_output_path = os.path.join(subfolder_path, 'geometry.png') 
        # Mettere "inputs.png" se si vuole il frame originale, "geometry.png" se si vuole il frame modificato
        if os.path.isfile(image_output_path):
            frame_files.append(image_output_path)

frame_files.sort(key = lambda x: int(x.split('/')[-2].split('_')[1]))
# x è il percorso del file, x.split('/')[-2] è il nome della cartella contenente il frame, 
# x.split('/')[-2].split('_')[1] è il numero del frame
# Si ordina la lista in base al numero del frame

# Si utilizza la prima immagine per ottenere le dimensioni del video
frame = cv2.imread(frame_files[0])
height, width, layers = frame.shape

video_path = '/andromeda/personal/fgastaldi/facemorphic_output/53189cf9202642bd917f31d6eabf8053/AU_51/frames_1/video2.avi' 
# Percorso del video da creare e rispettivo nome
video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))

for frame_file in frame_files:
    video.write(cv2.imread(frame_file))

video.release()
cv2.destroyAllWindows()

print('Video path:', video_path)


