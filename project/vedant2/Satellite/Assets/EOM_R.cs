using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EOM_R : MonoBehaviour
{

    public float torqueX, torqueY, torqueZ;
    public Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        float turnX = Input.GetAxis("Fire3");
        float turnY = Input.GetAxis("Horizontal");
        float turnZ = Input.GetAxis("Fire1");
        rb.AddTorque(transform.forward * torqueX * turnX * (1));
        rb.AddTorque(transform.up * torqueY * turnY * (1));
        rb.AddTorque(transform.right * torqueZ * turnZ * (1));
    }
}