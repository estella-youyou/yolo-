"""实践任务任务一（使用预训练模型yolo8n.pt进行图像检测）
完整图像演示，展示所有可获得的信息。"""
import os
from ultralytics import YOLO

class YOLOv8CompleteOutput:
    def __init__(self,model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        print(f"检测图片:{image_path}")
        results = self.model(image_path, verbose=False)
        return results
    
    def show_results(self, results):
        result = results[0] # 取图片的结果(results返回一个列表) 
        output_file = '检测结果.jpg'
        result.save(output_file)
        print(f" 结果已保存: {output_file}")
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"\n检测到 {len(result.boxes)} 个物体:")
            print("="*60)

            for i,box in enumerate(result.boxes):
                x1,y1,x2,y2 = box.xyxy[0].tolist()

                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                confidence = float(box.conf[0])

                print(f"{i+1:2d}. {class_name:15s} | 置信度: {confidence:.1%}")
                print(f"     位置: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                print(f"     宽高: {x2 - x1:.0f}x{y2 - y1:.0f}")
                print()   

        else:
            print("未检测到物体")    

if __name__ == "__main__":
    detector = YOLOv8CompleteOutput('yolov8n.pt')

    image_path = r"task1\task1.jpg"
    results = detector.detect(image_path)
    
    if results:
        detector.show_results(results)