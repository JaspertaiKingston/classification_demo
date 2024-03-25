from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential

def CaptionModel(
    image_file = None,
    endpoint = "https://computervision94327716.cognitiveservices.azure.com/",
    key = "c70e45249a274082b54f1dee53b94cbc"
):
    if image_file is None:
        return {"Caption": "no image"}
    
    client = ImageAnalysisClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(key)
    )
    result = client.analyze(
        image_data=image_file,
        visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ, VisualFeatures.OBJECTS],
        gender_neutral_caption=True,
        language='en'
    )
    if result.caption is not None:
        message = f"'{result.caption.text}'"
        confidence = f"{result.caption.confidence:.4f}"
        objects = result.objects.as_dict()
    return {"Caption": message, "Confidence": confidence, "Object": objects['values']}
