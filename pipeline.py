import kfp
from kfp import dsl
from kfp.components import OutputPath


def predict_object_op(image_list: str, input_prompt: str):
    return dsl.ContainerOp(
        name="predict object",
        image="registry.giaohangtietkiem.vn/vision-ai/base-image:v1.0.9",
        command=["python3", "main.py", image_list, input_prompt],
        file_outputs={
            'result_csv': '/tmp/result.csv' 
        }
    )

def pipeline(data: str, prompt: str) -> None :
    predict_object_task = predict_object_op(image_list=data, input_prompt= prompt).set_gpu_limit(1)
    rs_output = predict_object_task.output

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='pipeline.yaml')