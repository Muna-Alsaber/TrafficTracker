import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort 

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize SORT tracker  
sort_tracker = Sort(max_age=5, min_hits=3)  

# Load video and create video writer obj
cap = cv2.VideoCapture('56310-479197605_small.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Define lines for counting vehicles, the line in is green, and the line out is red
line_in = [(480, 300), (1000, 300)] 
line_out = [(0, 300), (480, 300)]   

# counters
count_in = 0
count_out = 0

# Dictionaries to store the last position of each track ID 
last_positions = {}

# Sets to store counted track IDs, I used set to to prevent duplicates in counting for each ID or obj
counted_in = set()
counted_out = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # detection
    results = model(frame)
    
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = model.names[int(box.cls)]
            score = box.conf.item()
            if label in ['car', 'truck'] and score > 0.4:  # Filter only cars and trucks with confidence > 0.4
                detections.append(([x1, y1, x2, y2], score, label))
                # Draw bounding box and label from YOLO
                center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                cv2.circle(frame, center, 5, (255, 255, 0), -1)
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'{label}', (x1 +60, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    detections_array = []
    for detection in detections:
        bbox, score, label = detection
        x1, y1, x2, y2 = bbox
        detections_array.append([x1, y1, x2, y2, score])

    # Convert detections_array to numpy array to match update function in Sort alogrithm 
    detections_np = np.array(detections_array)
    # Update SORT tracker with YOLO detections
    tracked_objects = sort_tracker.update(detections_np)
    
    for track in tracked_objects:
        x1, y1, x2, y2, track_id = track
    
        # Access track_id and other attributes
        track_id = int(track_id)
        #print(f"Track ID: {track_id}, Bounding Box: ({x1}, {y1}) to ({x2}, {y2})")
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        
        # Draw bounding box,center and truck id
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.circle(frame, center, 5, (125, 155, 0), -1)
        cv2.putText(frame, f'ID: {track_id}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Initialize last position if not present
        if track_id not in last_positions:
            last_positions[track_id] = center
        
        # Check if the vehicle crosses the line and has not been counted yet
        if track_id not in counted_in and last_positions[track_id][1] > line_in[0][1] and center[1] <= line_in[0][1]:
            count_in += 1
            counted_in.add(track_id)
            print(f'Vehicle {track_id} entered (Inbound)')
        
        if track_id not in counted_out and last_positions[track_id][1] < line_out[0][1] and center[1] >= line_out[0][1]:
            count_out += 1
            counted_out.add(track_id)
            print(f'Vehicle {track_id} exited (Outbound)')
        
        # Update last position
        last_positions[track_id] = center
        
    # Draw lines and Display counts
    cv2.line(frame, line_in[0], line_in[1], (0, 255, 0), 2)
    cv2.line(frame, line_out[0], line_out[1], (0, 0, 255), 2)
    
    cv2.putText(frame, f'In: {count_in}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Out: {count_out}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Write the frame to the output file
    out.write(frame)

    cv2.imshow('YOLOv8 + SORT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
