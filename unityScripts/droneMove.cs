using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class droneMove : MonoBehaviour
{
    public Rigidbody rb;
    public GameObject fR;
    public GameObject fL;
    public GameObject bR;
    public GameObject bL;

    public Vector3 _frontRightRotor, _frontLeftRotor, _backLeftRotor, _backRightRotor;
    [SerializeField]
    private float _frontRightSpeed, _frontLeftSpeed, _backRightSpeed, _backLeftSpeed, _upForce, _constForce;

    private void Start()
    {
        rb = this.GetComponent<Rigidbody>();

        _frontRightRotor = (fR.transform.position - rb.transform.position);
        _frontLeftRotor = (fL.transform.position - rb.transform.position);
        _backLeftRotor = (bL.transform.position - rb.transform.position);
        _backRightRotor = (bR.transform.position - rb.transform.position);
    }

    void FixedUpdate()
    {
        AddForceRotor(_constForce, (transform.rotation * _frontRightRotor) + transform.position);
        AddForceRotor(_constForce, (transform.rotation * _frontLeftRotor) + transform.position);
        AddForceRotor(_constForce, (transform.rotation * _backRightRotor) + transform.position);
        AddForceRotor(_constForce, (transform.rotation * _backLeftRotor) + transform.position);

        if (Input.GetKey("space"))
        {
            AddForceRotor(_upForce, (transform.rotation * _frontRightRotor) + transform.position);
            AddForceRotor(_upForce, (transform.rotation * _frontLeftRotor) + transform.position);
            AddForceRotor(_upForce, (transform.rotation * _backRightRotor) + transform.position);
            AddForceRotor(_upForce, (transform.rotation * _backLeftRotor) + transform.position);
        }

        if (Input.GetKey("w"))
        {
            AddForceRotor(_backRightSpeed, (transform.rotation * _backRightRotor) + transform.position);
            AddForceRotor(_backLeftSpeed, (transform.rotation * _backLeftRotor) + transform.position);
        }

        if (Input.GetKey("s"))
        {
            AddForceRotor(_frontRightSpeed, (transform.rotation * _frontRightRotor) + transform.position);
            AddForceRotor(_frontLeftSpeed, (transform.rotation * _frontLeftRotor) + transform.position);
        }

        if (Input.GetKey("a"))
        {
            AddForceRotor(_frontRightSpeed, (transform.rotation * _frontRightRotor) + transform.position);
            AddForceRotor(_backRightSpeed, (transform.rotation * _backRightRotor) + transform.position);
        }

        if (Input.GetKey("d"))
        {
            AddForceRotor(_frontLeftSpeed, (transform.rotation * _frontLeftRotor) + transform.position);
            AddForceRotor(_backLeftSpeed, (transform.rotation * _backLeftRotor) + transform.position);
        }

    }

    private void AddForceRotor(float speed, Vector3 blade)
    {
        rb.AddForceAtPosition(speed * transform.up * Time.deltaTime, blade);
    }

}