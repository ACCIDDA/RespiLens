import React from "react";

const NSSPView = () => {
  console.log(`You made it to nssp`);
  const containerStyle = {
    padding: "40px",
    textAlign: "center",
    fontFamily: "sans-serif",
  };

  const statusStyle = {
    color: "#666",
    fontStyle: "italic",
    marginTop: "10px",
  };

  return (
    <div style={containerStyle}>
      <h1>Welcome to NSSP View</h1>
      <p style={statusStyle}>It is currently under construction.</p>
    </div>
  );
};

export default NSSPView;
