using System.Collections;
using System.Collections.Generic;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Unity.Mathematics;
using UnityEngine;

[BurstCompile]
public struct PushData
{
    public Vector3 Position;
    public float Strength;
}

public class MeshDeformation : MonoBehaviour
{
    [SerializeField] private float m_DeformStrength;
    [SerializeField] private float m_DeformDistance;
    [SerializeField] private float m_SpringForce;
    [SerializeField] private float m_SpringDamping;

    private JobHandle m_DeformJobHandle;
    private Mesh m_Mesh;

    private Vector3[] m_OriginalVertices;
    private Vector3[] m_DeformedVertices;
    private Vector3[] m_VertexVelocities;

    private NativeArray<PushData> m_PushDatasIn;
    private NativeArray<float3> m_VertexVelocitiesIn;
    private NativeArray<float3> m_DeformedVerticesIn;
    private NativeArray<float3> m_OriginalVerticesIn;

    private List<PushData> m_PushData;

    float m_ScaleFactor;

    [BurstCompile]
    private struct MeshDeformJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<PushData> PushDatas;
        public NativeArray<float3> VertexVelocities;
        public NativeArray<float3> DeformedVertices;
        [ReadOnly] public NativeArray<float3> OriginalVertices;

        [ReadOnly] public float DeformStrength;
        [ReadOnly] public float ScaleFactor;
        [ReadOnly] public float SpringForce;
        [ReadOnly] public float SpringDamping;
        [ReadOnly] public float DeltaTime;

        public void PushVertex(int index, float3 point, float forceScalar)
        {
            float3 pointToVertex = DeformedVertices[index] - point;
            pointToVertex *= ScaleFactor;

            float adjustedForce = DeformStrength * forceScalar;
            adjustedForce /= adjustedForce + (1.0f + math.lengthsq(pointToVertex));
            float velocity = adjustedForce * DeltaTime;
            VertexVelocities[index] += math.normalize(pointToVertex) * velocity;
        }

        public void Execute(int index)
        {
            for (int i = 0; i < PushDatas.Length; i++)
            {
                PushVertex(index, PushDatas[i].Position, PushDatas[i].Strength);
            }

            float3 velocity = VertexVelocities[index];
            float3 displacement = DeformedVertices[index] - OriginalVertices[index];

            velocity -= displacement * SpringForce * DeltaTime;
            velocity *= 1.0f - SpringDamping * DeltaTime;
            VertexVelocities[index] = velocity;
            DeformedVertices[index] += velocity * DeltaTime;
        }
    }

    private void Start()
    {
        m_Mesh = GetComponent<MeshFilter>().mesh;
        m_PushData = new List<PushData>();

        m_OriginalVertices = m_Mesh.vertices;
        m_DeformedVertices = new Vector3[m_OriginalVertices.Length];
        m_VertexVelocities = new Vector3[m_OriginalVertices.Length];

        for (int i = 0; i < m_OriginalVertices.Length; i++)
        {
            m_DeformedVertices[i] = m_OriginalVertices[i];
        }

        m_OriginalVerticesIn = GetNativeVertexArrays(m_OriginalVertices);
    }

    private void Update()
    {
        m_ScaleFactor = transform.localScale.x;

        m_VertexVelocitiesIn = GetNativeVertexArrays(m_VertexVelocities);
        m_DeformedVerticesIn = GetNativeVertexArrays(m_DeformedVertices);

        m_PushDatasIn = new NativeArray<PushData>(m_PushData.Count, Allocator.TempJob, NativeArrayOptions.ClearMemory);
        for (int i = 0; i < m_PushData.Count; i++)
        {
            m_PushDatasIn[i] = m_PushData[i];
        }
        m_PushData.Clear();

        MeshDeformJob meshDeformJob = new MeshDeformJob()
        {
            VertexVelocities = m_VertexVelocitiesIn,
            DeformedVertices = m_DeformedVerticesIn,
            OriginalVertices = m_OriginalVerticesIn,
            PushDatas = m_PushDatasIn,
            DeformStrength = m_DeformStrength,
            ScaleFactor = m_ScaleFactor,
            SpringForce = m_SpringForce,
            SpringDamping = m_SpringDamping,
            DeltaTime = Time.deltaTime
        };

        m_DeformJobHandle = meshDeformJob.Schedule(m_DeformedVertices.Length, 32);

    }

    private void LateUpdate()
    {
        m_DeformJobHandle.Complete();

        SetNativeVertexArray(m_DeformedVertices, m_DeformedVerticesIn);
        SetNativeVertexArray(m_VertexVelocities, m_VertexVelocitiesIn);

        m_DeformedVerticesIn.Dispose();
        m_VertexVelocitiesIn.Dispose();

        m_Mesh.vertices = m_DeformedVertices;
        m_Mesh.RecalculateNormals();
    }

    public void DeformAtPoint(Vector3 point, float forceScalar)
    {
        point = transform.InverseTransformPoint(point);

        m_PushData.Add(new PushData
        {
            Position = point,
            Strength = forceScalar
        });
    }

    private void OnDestroy()
    {
#if !UNITY_EDITOR
        m_VertexVelocitiesIn.Dispose();
        m_DeformedVerticesIn.Dispose();
        m_OriginalVerticesIn.Dispose();
#endif
    }

    unsafe NativeArray<float3> GetNativeVertexArrays(Vector3[] vertexArray)
    {
        NativeArray<float3> verts = new NativeArray<float3>(vertexArray.Length, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        fixed (void* vertexBufferPointer = vertexArray)
        {
            UnsafeUtility.MemCpy(NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(verts),
                vertexBufferPointer, vertexArray.Length * (long)UnsafeUtility.SizeOf<float3>());
        }

        return verts;
    }

    unsafe void SetNativeVertexArray(Vector3[] vertexArray, NativeArray<float3> vertexBuffer)
    {
        fixed (void* vertexArrayPointer = vertexArray)
        {
            UnsafeUtility.MemCpy(vertexArrayPointer, NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(vertexBuffer), vertexArray.Length * (long)UnsafeUtility.SizeOf<float3>());
        }
    }
}
