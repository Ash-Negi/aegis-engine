package com.aegis.execution.signal;

/** A signal that fails validation. Dropped, never corrected. */
public class InvalidSignalException extends RuntimeException {

    public InvalidSignalException(String message) {
        super(message);
    }
}
