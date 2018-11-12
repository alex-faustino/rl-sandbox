using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EOM_Body : MonoBehaviour {

    public float torqueL_X,torqueR_X ,torqueL_Y, torqueR_Y, torqueL_Z, torqueR_Z;
    public Rigidbody rb;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        float turnR_X = Input.GetAxis("Fire3");
        float turnL_X = Input.GetAxis("Jump");

        float turnR_Y = Input.GetAxis("Horizontal");
        float turnL_Y = Input.GetAxis("Vertical");

        float turnR_Z = Input.GetAxis("Fire1");
        float turnL_Z = Input.GetAxis("Fire2");

        rb.AddTorque(transform.forward * torqueL_X * turnL_X * (-1));
        rb.AddTorque(transform.forward * torqueR_X * turnR_X * (-1));

        rb.AddTorque(transform.up * torqueL_Y * turnL_Y * (-1));
        rb.AddTorque(transform.up * torqueR_Y * turnR_Y*(-1));

        rb.AddTorque(transform.right * torqueL_Z * turnL_Z * (-1));
        rb.AddTorque(transform.right * torqueR_Z * turnR_Z * (-1));
    }
}