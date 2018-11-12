using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EOM_L : MonoBehaviour
{

    public float torqueX,torqueY, torqueZ;
    public Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        float turnX = Input.GetAxis("Jump");
        float turnY = Input.GetAxis("Vertical");
        float turnZ = Input.GetAxis("Fire2");
        rb.AddTorque(transform.forward * torqueX * turnX * (1));
        rb.AddTorque(transform.up * torqueY * turnY * (1));
        rb.AddTorque(transform.right * torqueZ * turnZ * (1));
    }
}