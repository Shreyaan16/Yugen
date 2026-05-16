from src.pipeline import InferencePipeline

pipeline = InferencePipeline()

print(pipeline.recommend_for_user(user_id=0, top_k=10))
print(pipeline.recommend_similar_to(title="Mobile Suit Gundam", n=5))
