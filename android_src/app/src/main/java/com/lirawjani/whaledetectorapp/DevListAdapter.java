package com.lirawjani.whaledetectorapp;

import android.annotation.SuppressLint;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.lirawjani.whaledetectorapp.R;

import java.util.ArrayList;
import java.util.function.Consumer;
public class DevListAdapter extends RecyclerView.Adapter<DevListAdapter.ViewHolder> {
    private final ArrayList<String> data;
    private Consumer<String> onClicked;

    public DevListAdapter(ArrayList<String> data, Consumer<String> onClicked) {
        super();
        this.data = data;
        this.onClicked = onClicked;
    }

    // Create new views (invoked by the layout manager)
    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(ViewGroup viewGroup, int viewType) {
        View view = LayoutInflater.from(viewGroup.getContext())
                .inflate(R.layout.dev_list_entry, viewGroup, false);
        return new ViewHolder(view);
    }

    // Replace the contents of a view (invoked by the layout manager)
    @Override
    public void onBindViewHolder(ViewHolder viewHolder, @SuppressLint("RecyclerView") final int position) {
        viewHolder.getButton().setText(data.get(position));
        viewHolder.getButton().setOnClickListener(v -> onClicked.accept(data.get(position)));
    }

    @Override
    public int getItemCount() {
        return data.size();
    }

    public static class ViewHolder extends RecyclerView.ViewHolder {
        private final Button button;

        public ViewHolder(View view) {
            super(view);
            button = view.findViewById(R.id.bt_dev_entry_button);
        }

        public Button getButton() {
            return button;
        }
    }

    public void setOnClicked(Consumer<String> onClicked) {
        this.onClicked = onClicked;
    }

    public void updateData(ArrayList<String> newData) {
        data.clear();
        data.addAll(newData);
        notifyDataSetChanged();
    }
}