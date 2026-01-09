fetch("members.json")
  .then(res => res.json())
  .then(data => {
    const container = document.getElementById("members-container");

    data.forEach(member => {
      const card = document.createElement("div");
      card.classList.add("member-card");

      card.innerHTML = `
        <img src="${member.photo}" alt="${member.name}">
        <h3>${member.name}</h3>
        <p><strong>Position:</strong> ${member.position}</p>
        <p><strong>Email:</strong> <a href="mailto:${member.email}">${member.email}</a></p>
		<p><strong>Bio:</strong> ${member.bio}</p>
      `;

      container.appendChild(card);
    });
  })
  .catch(err => console.error("Error loading members:", err));
