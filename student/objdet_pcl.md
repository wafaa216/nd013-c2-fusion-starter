
### What is happening in Step 5?

In lidar data, **intensity** tells us how reflective an object is. Some objects reflect a lot of light (like metal or glass), while others reflect very little (like wood or asphalt). These extreme values are called  **outliers** . If we don't handle these outliers, the image might look bad because most of the reflectivity will be very faint compared to the extreme values.

To fix this, Step 5:

1. **Focuses on the normal range** of reflectivity by ignoring very low and very high values.
2. **Scales** this range to fit within the range of 0-255 (which is how images are usually displayed).

---

### Breaking it Down:

#### 1.  **Percentiles (1% and 99%)** :

* The **1st percentile** is the value where the lowest 1% of data falls.
* The **99th percentile** is the value where the highest 1% of data falls.
* These percentiles ignore the extreme edges of the data, focusing on the "typical" range.

#### 2.  **Clipping** :

* Clipping means forcing any values below the 1st percentile to become the 1st percentile value.
* Similarly, any values above the 99th percentile are set to the 99th percentile value.
* This keeps everything within a reasonable range.

#### 3.  **Normalization to 8-Bit Scale** :

* After clipping, the intensity values are scaled to fit between 0 and 255 (the range for image brightness levels).
* The formula for scaling is:
  Normalized Value=Value−MinimumMaximum−Minimum×255\text{Normalized Value} = \frac{\text{Value} - \text{Minimum}}{\text{Maximum} - \text{Minimum}} \times 255**Normalized Value**=**Maximum**−**Minimum**Value**−**Minimum****×**255**
* This makes the data suitable for visualization.

---

### Why Do This?

Imagine you're looking at a grayscale image:

* Without this step: The image might look too dark or too bright because extreme values dominate.
* After this step: The image will have good contrast, and you can clearly see objects of varying reflectivity.


```python
# Step 1: Calculate the 1st and 99th percentiles of the intensity data
low_percentile = np.percentile(ri_intensity, 1)
high_percentile = np.percentile(ri_intensity, 99)

# Step 2: Clip the intensity values to the percentile range
ri_intensity = np.clip(ri_intensity, low_percentile, high_percentile)

# Step 3: Normalize the clipped values to the 0-255 range
ri_intensity = 255 * (ri_intensity - low_percentile) / (high_percentile - low_percentile)

# Step 4: Convert the normalized values to integers (0-255)
img_intensity = ri_intensity.astype(np.uint8)

```



### Analogy:

Think of this like adjusting the brightness and contrast of a photo. If some areas are extremely bright (like the sun) and others are very dark, you adjust the settings to make the photo look balanced and clear.
