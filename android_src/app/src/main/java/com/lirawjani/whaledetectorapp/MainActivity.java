package com.lirawjani.whaledetectorapp;

import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.util.ArrayList;
import java.util.List;

import okhttp3.MediaType;
import okhttp3.RequestBody;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import com.google.gson.JsonObject;

public class MainActivity extends AppCompatActivity {
    private static final String BASE_URL = "https://whale-detector.vercel.app/";  // IP address of Raspberry Pi
    //private static final String BASE_URL = "http://192.168.1.98:5000/";  // IP address of Raspberry Pi

    private ArrayList<String> data;
    private static final int REQUEST_BLUETOOTH_PERMISSIONS = 1;
    private String curDevice = "None";
    private final Handler handler = new Handler();
    private API api;
    private RecyclerView.LayoutManager layoutMgr;
    private RecyclerView rv;
    private DevListAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });
        Retrofit retrofit = new Retrofit.Builder()
                .baseUrl(BASE_URL)
                .addConverterFactory(GsonConverterFactory.create())
                .build();

        updateWhaleProb(0);
        api = retrofit.create(API.class);
        handler.post(UIUpdate);

        rv = findViewById(R.id.my_device_bt_connections);
        layoutMgr = new LinearLayoutManager(this);
        rv.setLayoutManager(layoutMgr);
        data = new ArrayList<>();

        adapter = new DevListAdapter(new ArrayList<>(), name -> {
            curDevice = name;
            Log.d("MainActivity", "Selected device: " + name);
        });

        rv.setAdapter(adapter);
    }

    Runnable UIUpdate = new Runnable() {
        @Override
        public void run() {
            if(!curDevice.equals("None")) getValue(curDevice);
            updateDevicesList();
            handler.postDelayed(this, 3000);  // Request every 5 seconds
        }
    };

    private JsonObject curData;
    private void getValue(String name)
    {
        //Log.d("MainActivity", "Getting value for device: " + curData);
        if (curData != null && curData.has(name))
        {
            JsonObject device = curData.get(name).getAsJsonObject();
            double prob = device.get("prob").getAsDouble();
            updateWhaleProb(prob);

            String loc = device.get("location").getAsString();
            String last_upd = device.get("last_upd").getAsString();

            TextView status = findViewById(R.id.text_bt_status_data);
            status.setText(
                    String.format("%s\n%s\n%s\n %.6f%%", name, loc, last_upd, prob * 100)
            );
        }
    }

    private void updateDevicesList() {
        api.getList().enqueue(new Callback<JsonObject>() {
            @Override
            public void onResponse(Call<JsonObject> call, Response<JsonObject> response) {
                if (response.isSuccessful()) {
                    JsonObject listResponse = response.body();
                    if (listResponse != null) {
                        Log.d("MainActivity", "Got list of devices: " + listResponse);
                        ArrayList<String> newData = new ArrayList<>(listResponse.keySet());
                        adapter.updateData(newData);
                        curData = listResponse;
                    }
                }
                else
                {
                    Log.e("MainActivity", "Error with fetching devices list");
                }
            }

            @Override
            public void onFailure(Call<JsonObject> call, Throwable t) {
                Log.e("MainActivity", "Error: " + t.getMessage());
            }
        });
    }

    private static final int LERP_DURATION = 2000; // Duration in milliseconds
    private final Handler whaleProbLerpHandler = new Handler();
    private double currentProb = 0.0;

    public void updateWhaleProb(double targetProb) {
        final ImageView progressWheel = findViewById(R.id.water_progress_bar);
        final WaterProgBar waterProgressWheel = (WaterProgBar) progressWheel.getDrawable();
        final TextView waterProb = findViewById(R.id.water_prob);

        long startTime = System.currentTimeMillis();
        whaleProbLerpHandler.post(new Runnable() {
            @Override
            public void run() {
                long elapsedTime = System.currentTimeMillis() - startTime;
                float t = Math.min(1, (float) elapsedTime / LERP_DURATION);
                currentProb = lerp(currentProb, targetProb, t);

                waterProgressWheel.setProgress(currentProb);
                waterProgressWheel.setColor(
                        ContextCompat.getColor(MainActivity.this, R.color.normal_blue),
                        ContextCompat.getColor(MainActivity.this, R.color.lighter_blue)
                );

                String pcnt = String.format("%.2f%%", currentProb * 100);
                //Log.d("MainActivity", "Updating whale probability to: " + String.format("%.6f%%", currentProb*100));
                waterProb.setText(pcnt);

                if (t < 1) {
                    whaleProbLerpHandler.post(this);
                }
            }
        });
    }

    private double lerp(double start, double end, float t) {
        return start + t * (end - start);
    }
}