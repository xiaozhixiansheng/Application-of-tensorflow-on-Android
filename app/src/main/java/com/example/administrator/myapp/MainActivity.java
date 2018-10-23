package com.example.administrator.myapp;

import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

public class MainActivity extends AppCompatActivity {

    static {
        System.loadLibrary("tensorflow_inference");
    }

    private TensorFlowInferenceInterface inferenceInterface = null;
    private static final String mode_file = "file:///android_asset/MnistTF_model.pb";
    private static final String INPUT_NODE = "conv2d_1_input_2:0"; //模型中输入层的名字
    private static final String OUTPUT_NODE = "dense_3_2/Softmax:0"; //模型中输出层的名字
    private static final int NUM_CLASSES = 10; //样本集的类别数量,mnist数据集对应10
    private float[] inputs_data = new float[784];
    private float[] outputs_data = new float[10];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Todo:
        TextView text = (TextView)findViewById(R.id.MyTextView);
        getPicturePixel();

        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), mode_file);
        //输入节点名称 输入数据 数据大小
        //填充数据 1, 28, 28, 1
        inferenceInterface.feed(INPUT_NODE, inputs_data, 1, 28, 28, 1);
        //运行神经网络
        inferenceInterface.run(new String[]{OUTPUT_NODE});
        //取出输出层的数据
        //输出层名称 输出数据
        inferenceInterface.fetch(OUTPUT_NODE, outputs_data);

        int logit = 0;
        //找出预测结果
        for (int i = 1; i < NUM_CLASSES; i++){
            if (outputs_data[i] > outputs_data[logit])
                logit = i;
        }
        text.setText("The number is " + logit);
    }

    private void getPicturePixel(){
        try{
            Resources res = getResources();
            Bitmap bitmap = BitmapFactory.decodeResource(res, R.mipmap.picture8);
            ImageView img = (ImageView)findViewById(R.id.MyImageView);
            img.setImageBitmap(bitmap);
            int width = bitmap.getWidth();
            int height = bitmap.getHeight();

            //保存所有像素的数组，图片 宽 * 高
            int[] pixels = new int[width * height];
            bitmap.getPixels(pixels, 0, width, 0, 0, width, height);
            for (int i = 0; i < pixels.length; i++){
                inputs_data[i] = (float)pixels[i];
            }
        }catch(Exception e){
            Log.d("tag", e.getMessage());
        }
    }
}
