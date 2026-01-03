import logging
import sys
import os

# Ensure backend module can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_slm():
    print("Testing SLM Engine...")
    try:
        from backend.inference import SLMEngine
        engine = SLMEngine()
        
        context = "The quick brown fox jumps over the lazy dog."
        question = "What does the fox do?"
        
        print("\nGenerating answer...")
        answer = engine.generate_answer(context, question)
        print(f"\nContext: {context}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        if "jumps" in answer.lower():
            print("\nSUCCESS: Model answered correctly.")
        else:
            print("\nWARNING: Model answer might be incorrect.")
            
    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_slm()
