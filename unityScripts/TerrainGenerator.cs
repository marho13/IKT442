using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class TerrainGenerator : MonoBehaviour {

    public int depth = 20;

    public int width = 256;
    public int bredth = 256;

    public float scale = 20f;

    public float offsetX = 100f;
    public float offsetY = 100f;

    void Start()
    {
        offsetX = Random.RandomRange(0f, 9999f);
        offsetY = Random.RandomRange(0f, 9999f);
    }

    void Update()
    {
        Terrain terrain = GetComponent<Terrain>();
        terrain.terrainData = GenerateTerrain(terrain.terrainData);

    }

    TerrainData GenerateTerrain(TerrainData terrainData)
    {
        terrainData.heightmapResolution = width + 1;

        terrainData.size = new Vector3(width, depth, bredth);
        terrainData.SetHeights(0, 0, GenerateHeights());

        return terrainData;

    }

    float[,] GenerateHeights()
    {
        float[,] heights = new float[width, bredth];
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < bredth; y++)
            {
                heights[x, y] = CalculateHeight(x, y);// Some Perlin Noise Value
            }
        }
        return heights;
    }

    float CalculateHeight(int x, int y)
    {
        float xCoord = (float)x / width * scale + offsetX;
        float yCoord = (float)y / bredth * scale + offsetY;

        return Mathf.PerlinNoise(xCoord, yCoord);

    }

}
