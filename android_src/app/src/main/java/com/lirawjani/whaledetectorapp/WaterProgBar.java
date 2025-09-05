package com.lirawjani.whaledetectorapp;

import android.graphics.Canvas;
import android.graphics.ColorFilter;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;

import androidx.annotation.Nullable;

public class WaterProgBar extends Drawable {
    private Paint paint;
    private double progress;
    private int fgColor, bgColor;

    public WaterProgBar() {
        super();
        paint = new Paint();
        paint.setStrokeWidth(90f);
        paint.setStyle(Paint.Style.STROKE);
        paint.setAntiAlias(true);
        this.progress = 0;
    }

    @Override
    public void draw(Canvas canvas) {
        Rect bounds = getBounds();
        float centerX = bounds.exactCenterX();
        float centerY = bounds.exactCenterY();
        float radius = (Math.min(centerX, centerY) - (paint.getStrokeWidth() / 2f)) / 1.25f;

        float sweepAngle = 360f;
        paint.setColor(bgColor);
        canvas.drawArc(centerX - radius, centerY - radius, centerX + radius, centerY + radius, -90f, sweepAngle, false, paint);

        sweepAngle = (int)((progress / 100f) * 360f);
        paint.setColor(fgColor);
        canvas.drawArc(centerX - radius, centerY - radius, centerX + radius, centerY + radius, -90f, sweepAngle, false, paint);
    }

    @Override
    public void setAlpha(int alpha) {
        // This method is required to implement the Drawable class, but is not used in this example
    }

    @Override
    public void setColorFilter(@Nullable ColorFilter colorFilter) {
        // This method is required to implement the Drawable class, but is not used in this example
    }

    @Override
    public int getOpacity() {
        // This method is required to implement the Drawable class, but is not used in this example
        return PixelFormat.TRANSPARENT;
    }

    public void setColor(int fg, int bg)
    {
        this.fgColor = fg;
        this.bgColor = bg;
    }

    public void setProgress(double progressInside) {
        this.progress = progressInside * 100;
        this.progress = Math.max(this.progress, 1f);
        invalidateSelf();
    }
}
